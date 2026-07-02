from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable
from typing import Any
from urllib.request import urlopen


class LocalRuntimeError(RuntimeError):
    """Local model runtime is not ready."""


class EndpointConflictError(LocalRuntimeError):
    """Health endpoint is alive, but serves a different model."""


class LocalRuntimeTimeoutError(LocalRuntimeError):
    """Local model runtime did not become ready before timeout."""


class LocalRuntimeManager:
    def __init__(
        self,
        *,
        request_json: Callable[[str, float], object] | None = None,
        launch_process: Callable[[list[str]], object] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._request_json = request_json or _request_json
        self._launch_process = launch_process or _launch_process
        self._sleep = sleep
        self._monotonic = monotonic

    def ensure_ready(self, spec: Any) -> None:
        if getattr(spec, "location", None) != "local":
            return

        runtime = getattr(spec, "runtime", None)
        health_url = getattr(runtime, "health_url", None) if runtime is not None else None
        if not health_url:
            raise LocalRuntimeError(f"Local model {spec.id!r} has no runtime.health_url")
        expected = (
            getattr(runtime, "expected_model_contains", None)
            if runtime is not None
            else None
        ) or getattr(spec, "provider_model", "")

        try:
            payload = self._request_json(str(health_url), 2.0)
            _raise_if_unexpected_model(
                payload,
                expected=str(expected),
                model_id=str(getattr(spec, "id", "unknown")),
                health_url=str(health_url),
            )
            return
        except EndpointConflictError:
            raise
        except Exception as initial_error:
            launch_command = (
                getattr(runtime, "launch_command", ())
                if runtime is not None
                else ()
            )
            if not launch_command:
                raise LocalRuntimeError(
                    f"Local model {spec.id!r} is not running and has no runtime.launch_command"
                ) from initial_error

        self._launch_process([str(part) for part in launch_command])
        timeout = float(getattr(runtime, "startup_timeout_seconds", 60.0))
        interval = float(getattr(runtime, "poll_interval_seconds", 1.0))
        deadline = self._monotonic() + timeout
        last_error: Exception | None = None

        while self._monotonic() <= deadline:
            try:
                payload = self._request_json(str(health_url), 2.0)
                _raise_if_unexpected_model(
                    payload,
                    expected=str(expected),
                    model_id=str(getattr(spec, "id", "unknown")),
                    health_url=str(health_url),
                )
                return
            except EndpointConflictError:
                raise
            except Exception as exc:
                last_error = exc
                self._sleep(interval)

        raise LocalRuntimeTimeoutError(
            f"Timed out waiting for local model {spec.id!r} at {health_url}"
        ) from last_error


def _request_json(url: str, timeout: float) -> object:
    with urlopen(url, timeout=timeout) as response:  # noqa: S310 - local/user-configured health URL
        return json.loads(response.read().decode("utf-8"))


def _launch_process(command: list[str]) -> object:
    return subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def _raise_if_unexpected_model(
    payload: object,
    *,
    expected: str,
    model_id: str,
    health_url: str,
) -> None:
    if not expected:
        return
    model_names = _model_names(payload)
    if any(expected in name for name in model_names):
        return
    raise EndpointConflictError(
        f"endpoint conflict for {model_id!r}: {health_url} is serving "
        f"{model_names or ['<no models>']}, expected model containing {expected!r}"
    )


def _model_names(payload: object) -> list[str]:
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, list):
            names: list[str] = []
            for item in data:
                if isinstance(item, dict):
                    value = item.get("id") or item.get("model")
                    if value is not None:
                        names.append(str(value))
                elif item is not None:
                    names.append(str(item))
            return names
        if "id" in payload:
            return [str(payload["id"])]
    if isinstance(payload, list):
        return [str(item) for item in payload]
    return []


__all__ = [
    "EndpointConflictError",
    "LocalRuntimeError",
    "LocalRuntimeManager",
    "LocalRuntimeTimeoutError",
]
