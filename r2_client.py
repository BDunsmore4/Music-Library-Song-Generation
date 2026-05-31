"""
Cloudflare R2 storage client (S3-compatible) for persisting generated audio.

Stores each generated mp3 under generated/<uuid>.mp3 and keeps a single
generated.json manifest object listing every generated track. The manifest is
read-modify-write; a process-level lock serialises appends so concurrent
/generate requests within one worker can't lose updates. This is safe under a
single gunicorn worker — do not scale to multiple workers without external
locking.
"""

import io
import json
import os
import threading
import uuid

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

R2_ACCOUNT_ID        = os.environ.get("R2_ACCOUNT_ID", "")
R2_ACCESS_KEY_ID     = os.environ.get("R2_ACCESS_KEY_ID", "")
R2_SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY", "")
R2_BUCKET            = os.environ.get("R2_BUCKET", "")
# Public base URL (no trailing slash), e.g. https://pub-xxxx.r2.dev
R2_PUBLIC_BASE_URL   = os.environ.get("R2_PUBLIC_BASE_URL", "").rstrip("/")

MANIFEST_KEY = "generated.json"

_manifest_lock = threading.Lock()
_client = None


def _get_client():
    """Lazily build the boto3 S3 client pointed at the R2 endpoint."""
    global _client
    if _client is None:
        if not (R2_ACCOUNT_ID and R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY and R2_BUCKET):
            raise RuntimeError("R2 is not configured (missing R2_* env vars)")
        _client = boto3.client(
            "s3",
            endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
            aws_access_key_id=R2_ACCESS_KEY_ID,
            aws_secret_access_key=R2_SECRET_ACCESS_KEY,
            region_name="auto",
            config=Config(signature_version="s3v4"),
        )
    return _client


def upload_mp3(audio_bytes, key=None):
    """Upload mp3 bytes to R2 and return the public URL."""
    if key is None:
        key = f"generated/{uuid.uuid4().hex}.mp3"
    _get_client().put_object(
        Bucket=R2_BUCKET,
        Key=key,
        Body=audio_bytes,
        ContentType="audio/mpeg",
    )
    return f"{R2_PUBLIC_BASE_URL}/{key}"


def read_manifest():
    """Return the list of generated-track records, or [] if none yet."""
    try:
        obj = _get_client().get_object(Bucket=R2_BUCKET, Key=MANIFEST_KEY)
        return json.loads(obj["Body"].read())
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
            return []
        raise


def write_manifest(records):
    """Overwrite the manifest with the given list of records."""
    _get_client().put_object(
        Bucket=R2_BUCKET,
        Key=MANIFEST_KEY,
        Body=json.dumps(records).encode("utf-8"),
        ContentType="application/json",
    )


def append_to_manifest(record):
    """Append one record to the manifest atomically (within this process)."""
    with _manifest_lock:
        records = read_manifest()
        records.append(record)
        write_manifest(records)
        return records
