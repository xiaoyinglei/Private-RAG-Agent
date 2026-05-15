from __future__ import annotations

from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast


class S3ObjectStore:
    def __init__(
        self,
        *,
        bucket: str,
        endpoint_url: str | None = None,
        prefix: str = "",
        region_name: str | None = None,
        access_key_id: str | None = None,
        secret_access_key: str | None = None,
        session_token: str | None = None,
        client: object | None = None,
    ) -> None:
        self._bucket = bucket
        self._endpoint_url = endpoint_url
        self._prefix = prefix.strip("/")
        self._region_name = region_name
        self._access_key_id = access_key_id
        self._secret_access_key = secret_access_key
        self._session_token = session_token
        self._client: Any | None = cast(Any | None, client)
        self._cache_dir = TemporaryDirectory(prefix="rag-object-cache-")

    def put_bytes(self, content: bytes, *, suffix: str = "") -> str:
        digest = sha256(content).hexdigest()
        safe_suffix = suffix if suffix.startswith(".") or not suffix else f".{suffix}"
        key = f"{digest}{safe_suffix}"
        object_key = self._object_key(key)
        if not self.exists(key):
            self._client_instance().put_object(Bucket=self._bucket, Key=object_key, Body=content)
        return key

    def read_bytes(self, key: str) -> bytes:
        response = self._client_instance().get_object(Bucket=self._bucket, Key=self._object_key(key))
        body = cast(Any, response["Body"])
        return cast(bytes, body.read())

    def read_byte_range(self, key: str, start: int, end: int) -> bytes:
        if end <= start:
            return b""
        response = self._client_instance().get_object(
            Bucket=self._bucket,
            Key=self._object_key(key),
            Range=f"bytes={max(start, 0)}-{max(end - 1, 0)}",
        )
        body = cast(Any, response["Body"])
        return cast(bytes, body.read())

    def exists(self, key: str) -> bool:
        try:
            self._client_instance().head_object(Bucket=self._bucket, Key=self._object_key(key))
        except Exception as exc:
            error_code = self._error_code(exc)
            if error_code in {"404", "NoSuchKey", "NotFound"}:
                return False
            raise
        return True

    def path_for_key(self, key: str) -> Path:
        cache_path = Path(self._cache_dir.name) / key
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if not cache_path.exists():
            cache_path.write_bytes(self.read_bytes(key))
        return cache_path

    def close(self) -> None:
        self._cache_dir.cleanup()

    def _client_instance(self) -> Any:
        if self._client is None:
            import boto3

            self._client = boto3.client(
                "s3",
                endpoint_url=self._endpoint_url,
                region_name=self._region_name,
                aws_access_key_id=self._access_key_id,
                aws_secret_access_key=self._secret_access_key,
                aws_session_token=self._session_token,
            )
        return self._client

    def _object_key(self, key: str) -> str:
        if not self._prefix:
            return key
        return f"{self._prefix}/{key}"

    @staticmethod
    def _error_code(exc: Exception) -> str:
        response = getattr(exc, "response", None)
        error = response.get("Error", {}) if isinstance(response, dict) else {}
        code = error.get("Code")
        return str(code) if code is not None else exc.__class__.__name__
