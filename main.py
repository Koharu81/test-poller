from __future__ import annotations

import asyncio
import json
import os
import random
import signal
import sys
import time
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Any

import aiohttp
from loguru import logger
from pydantic import BaseModel, Field, field_validator


logger.remove()
logger.add(sys.stderr, serialize=True, level="INFO")


class JobStatusException(Exception):
    def __init__(self, message: str, job_id: str | None = None) -> None:
        self.job_id = job_id
        super().__init__(message)


class JobTimeoutException(JobStatusException):
    pass


class JobFailedException(JobStatusException):
    def __init__(self, message: str, job_id: str | None = None, conclusion: str | None = None) -> None:
        self.conclusion = conclusion
        super().__init__(message, job_id)


class JobConclusion(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    NEUTRAL = "neutral"


class JobRequestDTO(BaseModel):
    job_name: str = Field(min_length=1, max_length=200)
    parameters: dict[str, Any] = Field(default_factory=dict)
    ref: str = Field(default="main", min_length=1)

    @field_validator("job_name")
    @classmethod
    def validate_job_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("job_name must not be empty")
        return value.strip()


class JobStatusDTO(BaseModel):
    job_id: str
    status: str
    conclusion: str | None = None
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    updated_at: datetime
    raw_payload: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_completed(self) -> bool:
        return self.status == "completed"

    @property
    def is_successful(self) -> bool:
        return self.is_completed and self.conclusion == JobConclusion.SUCCESS.value


class RetryConfig(BaseModel):
    max_retries: int = Field(default=5, ge=0)
    base_delay_seconds: float = Field(default=1.0, gt=0)
    max_delay_seconds: float = Field(default=60.0, gt=0)
    jitter_seconds: float = Field(default=0.5, ge=0)


async def with_exponential_backoff(
    coro_fn,
    retry_config: RetryConfig,
    retryable_exceptions: tuple[type[BaseException], ...] = (aiohttp.ClientError, asyncio.TimeoutError),
):
    attempt = 0
    while True:
        try:
            return await coro_fn()
        except retryable_exceptions as exc:
            attempt += 1
            if attempt > retry_config.max_retries:
                logger.bind(attempt=attempt, error=str(exc)).error("retry_exhausted")
                raise
            delay = min(retry_config.base_delay_seconds * (2 ** (attempt - 1)), retry_config.max_delay_seconds)
            delay += random.uniform(0, retry_config.jitter_seconds)
            logger.bind(attempt=attempt, delay=round(delay, 3), error=str(exc)).warning("retrying_after_backoff")
            await asyncio.sleep(delay)


class AbstractJobStatusAdapter(ABC):
    def __init__(self, session: aiohttp.ClientSession, retry_config: RetryConfig | None = None) -> None:
        self._session = session
        self._retry_config = retry_config or RetryConfig()

    @abstractmethod
    async def create_job(self, request: JobRequestDTO) -> str:
        raise NotImplementedError

    @abstractmethod
    async def get_job_status(self, job_id: str) -> JobStatusDTO:
        raise NotImplementedError

    @abstractmethod
    async def close(self) -> None:
        raise NotImplementedError


class GitHubWorkflowAdapter(AbstractJobStatusAdapter):
    API_BASE = "https://api.github.com"

    def __init__(
        self,
        session: aiohttp.ClientSession,
        owner: str,
        repo: str,
        workflow_file: str,
        token: str,
        retry_config: RetryConfig | None = None,
    ) -> None:
        super().__init__(session, retry_config)
        self._owner = owner
        self._repo = repo
        self._workflow_file = workflow_file
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        self._dispatch_timestamp: float = 0.0

    async def create_job(self, request: JobRequestDTO) -> str:
        url = f"{self.API_BASE}/repos/{self._owner}/{self._repo}/actions/workflows/{self._workflow_file}/dispatches"
        payload = {"ref": request.ref, "inputs": request.parameters}

        async def _dispatch() -> None:
            async with self._session.post(url, headers=self._headers, json=payload) as response:
                if response.status not in (201, 204):
                    body = await response.text()
                    raise aiohttp.ClientResponseError(
                        request_info=response.request_info,
                        history=response.history,
                        status=response.status,
                        message=body,
                    )

        self._dispatch_timestamp = time.time()
        await with_exponential_backoff(_dispatch, self._retry_config)

        job_id = await self._resolve_run_id(request.job_name)
        logger.bind(job_id=job_id, job_name=request.job_name).info("job_created")
        return job_id

    async def _resolve_run_id(self, job_name: str, resolve_timeout_seconds: float = 30.0) -> str:
        url = f"{self.API_BASE}/repos/{self._owner}/{self._repo}/actions/workflows/{self._workflow_file}/runs"
        params = {"event": "workflow_dispatch", "per_page": "10"}
        deadline = time.time() + resolve_timeout_seconds

        while time.time() < deadline:
            async def _list_runs() -> dict[str, Any]:
                async with self._session.get(url, headers=self._headers, params=params) as response:
                    response.raise_for_status()
                    return await response.json()

            data = await with_exponential_backoff(_list_runs, self._retry_config)
            for run in data.get("workflow_runs", []):
                created_at = datetime.fromisoformat(run["created_at"].replace("Z", "+00:00")).timestamp()
                if created_at >= self._dispatch_timestamp - 2:
                    return str(run["id"])
            await asyncio.sleep(2.0)

        raise JobStatusException(
            f"workflow run for '{job_name}' could not be resolved within {resolve_timeout_seconds}s"
        )

    async def get_job_status(self, job_id: str) -> JobStatusDTO:
        url = f"{self.API_BASE}/repos/{self._owner}/{self._repo}/actions/runs/{job_id}"

        async def _get_run() -> dict[str, Any]:
            async with self._session.get(url, headers=self._headers) as response:
                response.raise_for_status()
                return await response.json()

        data = await with_exponential_backoff(_get_run, self._retry_config)

        status = data.get("status", "unknown")
        progress = 1.0 if status == "completed" else 0.5 if status == "in_progress" else 0.0

        return JobStatusDTO(
            job_id=job_id,
            status=status,
            conclusion=data.get("conclusion"),
            progress=progress,
            updated_at=datetime.fromisoformat(data["updated_at"].replace("Z", "+00:00")),
            raw_payload=data,
        )

    async def close(self) -> None:
        return None


class JobAdapterFactory:
    _registry: dict[str, type[AbstractJobStatusAdapter]] = {}

    @classmethod
    def register(cls, adapter_type: str, adapter_cls: type[AbstractJobStatusAdapter]) -> None:
        cls._registry[adapter_type] = adapter_cls

    @classmethod
    def create(cls, adapter_type: str, session: aiohttp.ClientSession, **kwargs: Any) -> AbstractJobStatusAdapter:
        adapter_cls = cls._registry.get(adapter_type)
        if adapter_cls is None:
            raise ValueError(f"unknown adapter type: {adapter_type}")
        return adapter_cls(session=session, **kwargs)


JobAdapterFactory.register("github_workflow", GitHubWorkflowAdapter)


class PollingConfig(BaseModel):
    interval_seconds: float = Field(default=5.0, gt=0)
    jitter_seconds: float = Field(default=1.5, ge=0)
    timeout_seconds: float = Field(default=600.0, gt=0)


async def poll_job_status(
    adapter: AbstractJobStatusAdapter,
    job_id: str,
    polling_config: PollingConfig | None = None,
) -> JobStatusDTO:
    config = polling_config or PollingConfig()
    deadline = time.monotonic() + config.timeout_seconds

    while True:
        status_dto = await adapter.get_job_status(job_id)
        logger.bind(job_id=job_id, status=status_dto.status, conclusion=status_dto.conclusion).info(
            "job_status_polled"
        )

        if status_dto.is_completed:
            if status_dto.is_successful:
                return status_dto
            raise JobFailedException(
                f"job {job_id} finished with conclusion '{status_dto.conclusion}'",
                job_id=job_id,
                conclusion=status_dto.conclusion,
            )

        if time.monotonic() >= deadline:
            raise JobTimeoutException(f"job {job_id} did not complete within {config.timeout_seconds}s", job_id=job_id)

        remaining = deadline - time.monotonic()
        wait_seconds = min(config.interval_seconds, max(remaining, 0.0))
        wait_seconds += random.uniform(0, config.jitter_seconds)
        await asyncio.sleep(wait_seconds)


class ShutdownController:
    def __init__(self) -> None:
        self._shutdown_event = asyncio.Event()

    def request_shutdown(self, *_: Any) -> None:
        self._shutdown_event.set()

    async def wait(self) -> None:
        await self._shutdown_event.wait()

    @property
    def is_shutting_down(self) -> bool:
        return self._shutdown_event.is_set()


def build_runtime_config_from_env() -> dict[str, Any]:
    return {
        "owner": os.environ.get("GITHUB_OWNER", ""),
        "repo": os.environ.get("GITHUB_REPO", ""),
        "workflow_file": os.environ.get("GITHUB_WORKFLOW_FILE", ""),
        "token": os.environ.get("GITHUB_TOKEN", ""),
        "job_name": os.environ.get("JOB_NAME", "manual-dispatch"),
        "ref": os.environ.get("GITHUB_REF", "main"),
        "timeout_seconds": float(os.environ.get("POLL_TIMEOUT_SECONDS", "600")),
        "interval_seconds": float(os.environ.get("POLL_INTERVAL_SECONDS", "5")),
    }


async def run_job_pipeline(runtime_config: dict[str, Any], shutdown: ShutdownController) -> dict[str, Any]:
    connector = aiohttp.TCPConnector(limit=100)
    timeout = aiohttp.ClientTimeout(total=30)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        adapter = JobAdapterFactory.create(
            "github_workflow",
            session=session,
            owner=runtime_config["owner"],
            repo=runtime_config["repo"],
            workflow_file=runtime_config["workflow_file"],
            token=runtime_config["token"],
        )

        try:
            request_dto = JobRequestDTO(job_name=runtime_config["job_name"], ref=runtime_config["ref"])
            job_id = await adapter.create_job(request_dto)

            polling_config = PollingConfig(
                interval_seconds=runtime_config["interval_seconds"],
                timeout_seconds=runtime_config["timeout_seconds"],
            )

            poll_task = asyncio.create_task(poll_job_status(adapter, job_id, polling_config))
            shutdown_task = asyncio.create_task(shutdown.wait())

            done, pending = await asyncio.wait({poll_task, shutdown_task}, return_when=asyncio.FIRST_COMPLETED)

            for task in pending:
                task.cancel()

            if shutdown_task in done:
                poll_task.cancel()
                return {
                    "job_id": job_id,
                    "status": "cancelled",
                    "conclusion": None,
                    "message": "shutdown requested before completion",
                }

            status_dto = poll_task.result()
            return {
                "job_id": status_dto.job_id,
                "status": status_dto.status,
                "conclusion": status_dto.conclusion,
                "updated_at": status_dto.updated_at.isoformat(),
            }

        except JobTimeoutException as exc:
            logger.bind(job_id=exc.job_id).error("job_timeout")
            return {"job_id": exc.job_id, "status": "timeout", "conclusion": None, "error": str(exc)}

        except JobFailedException as exc:
            logger.bind(job_id=exc.job_id, conclusion=exc.conclusion).error("job_failed")
            return {"job_id": exc.job_id, "status": "failed", "conclusion": exc.conclusion, "error": str(exc)}

        finally:
            await adapter.close()


async def main() -> None:
    shutdown = ShutdownController()
    loop = asyncio.get_running_loop()

    for sig_name in ("SIGINT", "SIGTERM"):
        sig = getattr(signal, sig_name, None)
        if sig is not None:
            try:
                loop.add_signal_handler(sig, shutdown.request_shutdown)
            except NotImplementedError:
                signal.signal(sig, lambda *_: shutdown.request_shutdown())

    runtime_config = build_runtime_config_from_env()
    result = await run_job_pipeline(runtime_config, shutdown)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
