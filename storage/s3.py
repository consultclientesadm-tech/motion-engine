"""S3/R2 (API S3) + lifecycle.

Layout das chaves:
  jobs/{job_id}/source.mp4            vídeo de entrada normalizado
  jobs/{job_id}/analysis/tracks.json  tracks + poses
  jobs/{job_id}/characters/{id}.glb   avatar
  jobs/{job_id}/audio/...             vozes e trilha final
  tmp/{job_id}/blocks/{i}/...         blocos intermediários (expiram pelo lifecycle)
  jobs/{job_id}/output.mp4            resultado
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import urllib.request
from functools import lru_cache
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from settings import get_settings

log = logging.getLogger(__name__)


@lru_cache
def client():
    s = get_settings()
    return boto3.client(
        "s3",
        endpoint_url=s.s3_endpoint or None,
        aws_access_key_id=s.s3_access_key or None,
        aws_secret_access_key=s.s3_secret_key or None,
        region_name=s.s3_region,
        config=Config(retries={"max_attempts": 5, "mode": "adaptive"}, signature_version="s3v4"),
    )


def bucket() -> str:
    return get_settings().s3_bucket


def job_key(job_id: str, *parts: str) -> str:
    return "/".join(["jobs", job_id, *parts])


def tmp_key(job_id: str, *parts: str) -> str:
    return "/".join(["tmp", job_id, *parts])


def upload(path: str | Path, key: str, content_type: str | None = None) -> str:
    extra = {"ContentType": content_type} if content_type else None
    client().upload_file(str(path), bucket(), key, ExtraArgs=extra)
    return key


def upload_bytes(data: bytes, key: str, content_type: str = "application/octet-stream") -> str:
    client().put_object(Bucket=bucket(), Key=key, Body=data, ContentType=content_type)
    return key


def download(key: str, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    client().download_file(bucket(), key, str(path))
    return path


def exists(key: str | None) -> bool:
    if not key:
        return False
    try:
        client().head_object(Bucket=bucket(), Key=key)
        return True
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def put_json(obj, key: str) -> str:
    return upload_bytes(json.dumps(obj, separators=(",", ":")).encode(), key, "application/json")


def get_json(key: str):
    return json.loads(client().get_object(Bucket=bucket(), Key=key)["Body"].read())


def presigned_url(key: str, expires: int = 24 * 3600) -> str:
    return client().generate_presigned_url("get_object", Params={"Bucket": bucket(), "Key": key},
                                           ExpiresIn=expires)


def fetch(url: str, dest: str | Path) -> Path:
    """Baixa `s3://bucket/key`, `key` do nosso bucket ou http(s) para `dest`."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if url.startswith("s3://"):
        b, _, key = url[5:].partition("/")
        client().download_file(b, key, str(dest))
    elif url.startswith(("http://", "https://")):
        with urllib.request.urlopen(url, timeout=60) as r, open(dest, "wb") as f:
            shutil.copyfileobj(r, f, length=1 << 20)
    elif os.path.exists(url):
        shutil.copyfile(url, dest)
    else:
        download(url, dest)
    return dest


def delete_prefix(prefix: str) -> int:
    n = 0
    paginator = client().get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket(), Prefix=prefix):
        objs = [{"Key": o["Key"]} for o in page.get("Contents", [])]
        if objs:
            client().delete_objects(Bucket=bucket(), Delete={"Objects": objs})
            n += len(objs)
    return n


def ensure_bucket_and_lifecycle() -> None:
    """Cria o bucket se faltar e aplica: tmp/ expira em N dias; uploads
    multipart incompletos são abortados em 1 dia (evita custo fantasma)."""
    s = get_settings()
    c = client()
    try:
        c.head_bucket(Bucket=s.s3_bucket)
    except ClientError:
        c.create_bucket(Bucket=s.s3_bucket)
    try:
        _put_lifecycle(c, s)
    except ClientError as e:
        # S3 local (rclone serve s3, moto) pode não implementar lifecycle: segue sem ele
        code = e.response.get("Error", {}).get("Code")
        if code not in ("NotImplemented", "MethodNotAllowed", "InvalidRequest", "501"):
            raise
        log.warning("S3 sem suporte a lifecycle (%s): tmp/ não expira sozinho", code)


def _put_lifecycle(c, s) -> None:
    c.put_bucket_lifecycle_configuration(
        Bucket=s.s3_bucket,
        LifecycleConfiguration={"Rules": [
            {"ID": "expire-tmp", "Status": "Enabled", "Filter": {"Prefix": "tmp/"},
             "Expiration": {"Days": s.s3_tmp_retention_days}},
            {"ID": "abort-multipart", "Status": "Enabled", "Filter": {"Prefix": ""},
             "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 1}},
        ]},
    )
