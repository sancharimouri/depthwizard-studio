"""Relative depth (DAv2-Small) via an inference host (docs/DEPLOY.md).

The host comes from the environment, read on every request:

  DAV2_INFERENCE_URL   default: the ZeroGPU Space "sancharimouri/DepthWizard2".
                       Either a Hugging Face Space id ("owner/name", or its
                       huggingface.co/spaces/... URL), called through the Gradio API
                       (gradio_client, api_name="/predict"), or a plain http(s) base URL
                       serving POST /predict (the desktop app's local ONNX server,
                       bridge/dav2_server_onnx.py).

  DW2_MODEL_URL        optional: the Cloud Run model service (docker/model/app.py). When set it is called FIRST,
                       with an ID token for this service's own service account (metadata server); on a
                       timeout, an unreachable host or a 5xx it falls back to the host above. The response
                       says which one answered ("host", plus "fallback" when the Space stood in).

Space calls always carry HF_TOKEN as an explicit Authorization header. gradio_client's
token= alone only authenticates the huggingface.co API calls, not the requests to
*.hf.space (checked 2026-09-26), which would make them anonymous for ZeroGPU quota.
Every request to a Space host is audited (auth_audit(), shown in /api/depth/status).

The response is the host's JSON, plus "host": 518x518 relative depth (larger = nearer),
NOT elevation, "u16-zlib" encoded (older hosts: raw float32 with no "encoding" field).
"""

from __future__ import annotations

import asyncio
import logging
import concurrent.futures
import os
import re
import tempfile
import threading
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import httpx  # noqa: E402
from fastapi import APIRouter, File, HTTPException, UploadFile  # noqa: E402

from backend.input import store  # noqa: E402
from backend.library import catalog  # noqa: E402
from backend.storage import library_store  # noqa: E402
from backend.storage.hf_checkpoints import token as hf_token  # noqa: E402

router = APIRouter(prefix="/api/depth")
TIMEOUT_S = float(os.environ.get("DAV2_TIMEOUT_S", "120"))
DEFAULT_SPACE = "sancharimouri/DepthWizard2"
_SPACE_URL = re.compile(r"^https?://huggingface\.co/spaces/([\w.-]+/[\w.-]+)/?$")
_SPACE_ID = re.compile(r"[\w.-]+/[\w.-]+")


# ------------------------------------------------------------------ host config
def space_id(host: str) -> str | None:
    """"owner/name" for a Space host, None for a plain http(s) URL."""
    if host.startswith(("http://", "https://")):
        m = _SPACE_URL.match(host)
        return m.group(1) if m else None
    return host if _SPACE_ID.fullmatch(host) else None


def host() -> str:
    return os.environ.get("DAV2_INFERENCE_URL", "").strip().rstrip("/") or DEFAULT_SPACE


# ------------------------------------------------------------------ auth audit
_audit = {"with_auth": 0, "without_auth": 0}
_audit_lock = threading.Lock()


def _record(request: httpx.Request) -> None:
    if request.url.host.endswith(".hf.space"):
        key = "with_auth" if request.headers.get("authorization", "").startswith("Bearer ") else "without_auth"
        with _audit_lock:
            _audit[key] += 1


# HF's edge in front of *.hf.space answers a share of requests with 502 before they reach any replica: no
# x-proxied-replica header (2026-09-29: 8-23% of requests, on this Space and on unrelated popular ones). Such a
# request never reached the app, so it is re-sent after these delays; the app's own errors are left alone.
EDGE_RETRY_DELAYS_S = (0.5, 1.0, 2.0, 3.0)


def _edge_502(request: httpx.Request, response: httpx.Response) -> bool:
    return (request.url.host.endswith(".hf.space") and response.status_code in (502, 503, 504)
            and "x-proxied-replica" not in response.headers)


def _install_audit() -> None:
    """Count every httpx request to *.hf.space and whether it carried a Bearer token; re-send edge 502s."""
    if getattr(httpx.Client.send, "_dw2_audit", False):
        return
    sync_send, async_send = httpx.Client.send, httpx.AsyncClient.send

    def send(self, request, **kw):
        for delay in (*EDGE_RETRY_DELAYS_S, None):
            _record(request)
            response = sync_send(self, request, **kw)
            if delay is None or not _edge_502(request, response):
                return response
            response.close()
            time.sleep(delay)

    async def asend(self, request, **kw):
        for delay in (*EDGE_RETRY_DELAYS_S, None):
            _record(request)
            response = await async_send(self, request, **kw)
            if delay is None or not _edge_502(request, response):
                return response
            await response.aclose()
            await asyncio.sleep(delay)

    send._dw2_audit = asend._dw2_audit = True
    httpx.Client.send, httpx.AsyncClient.send = send, asend


def auth_audit() -> dict:
    with _audit_lock:
        return dict(_audit)


_install_audit()


# ------------------------------------------------------------------ Space (Gradio API)
_clients: dict[str, object] = {}
_clients_lock = threading.Lock()


def _space_client(sid: str):
    tok = hf_token()
    if not tok:
        raise HTTPException(status_code=503, detail="HF_TOKEN is not set; Space calls are never made anonymously.")
    with _clients_lock:
        if sid not in _clients:
            from gradio_client import Client
            _clients[sid] = Client(sid, token=tok, headers={"Authorization": f"Bearer {tok}"},
                                   verbose=False, analytics_enabled=False)
        return _clients[sid]


def _space_predict(sid: str, body: bytes, suffix: str) -> dict:
    from gradio_client import handle_file
    client = _space_client(sid)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / f"input{suffix}"
        path.write_bytes(body)
        return client.predict(handle_file(str(path)), api_name="/predict")


# A Space that is restarting answers 502/503/504 for a short while (its Gradio SSR proxy is up
# before the Python app behind it; seen 2026-09-28), and an upload can be cut off mid-stream. Such
# blips are retried after these delays, each time with a fresh gradio Client (the cached one may
# hold the old instance's config). Quota errors and timeouts are never retried.
SPACE_RETRY_DELAYS_S = (2.0, 5.0)
_TRANSIENT = re.compile(r"\b50[234]\b|bad gateway|service unavailable|gateway time-?out|clientdisconnect|"
                        r"connection (?:reset|refused|aborted)|remote ?protocol|server disconnected|"
                        r"could not fetch (?:api info|config)", re.I)


def _is_transient(msg: str) -> bool:
    return "quota" not in msg.lower() and bool(_TRANSIENT.search(msg))


async def _call_space(sid: str, filename: str, body: bytes) -> dict:
    for attempt in range(len(SPACE_RETRY_DELAYS_S) + 1):
        try:
            return await asyncio.wait_for(asyncio.to_thread(_space_predict, sid, body, Path(filename).suffix or ".jpg"),
                                          timeout=TIMEOUT_S)
        except HTTPException:
            raise
        except asyncio.TimeoutError as exc:
            raise HTTPException(status_code=504, detail=f"Space {sid} timed out after {TIMEOUT_S:.0f}s.") from exc
        except (asyncio.CancelledError, concurrent.futures.CancelledError) as exc:
            task = asyncio.current_task()
            if task is not None and task.cancelling():
                raise  # this request itself is being cancelled (client gone, shutdown)
            # gradio_client cancels its job when the event stream breaks, e.g. HF's edge sends an HTML error page
            # into it (Render, 2026-09-29). A BaseException, so it used to escape as a bare 500.
            msg = "the Space's event stream broke (bad gateway) and the job was cancelled"
            if attempt < len(SPACE_RETRY_DELAYS_S):
                with _clients_lock:
                    _clients.pop(sid, None)
                await asyncio.sleep(SPACE_RETRY_DELAYS_S[attempt])
                continue
            raise HTTPException(status_code=502, detail=f"Space {sid}: {msg}") from exc
        except Exception as exc:  # noqa: BLE001 - gradio AppError carries ZeroGPU quota messages
            msg = str(exc).strip() or type(exc).__name__
            if attempt < len(SPACE_RETRY_DELAYS_S) and _is_transient(msg):
                with _clients_lock:
                    _clients.pop(sid, None)
                await asyncio.sleep(SPACE_RETRY_DELAYS_S[attempt])
                continue
            code = 429 if "quota" in msg.lower() else 502
            raise HTTPException(status_code=code, detail=f"Space {sid}: {msg[:300]}") from exc
    raise AssertionError("unreachable")


# ------------------------------------------------------------------ plain HTTP (desktop app's local server)
async def _call_http(url: str, filename: str, body: bytes, content_type: str, headers: dict | None = None,
                     path: str = "/predict", data: dict | None = None) -> dict:
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
            r = await client.post(f"{url}{path}", files={"file": (filename, body, content_type)}, data=data,
                                  headers=headers)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Inference host unreachable ({type(exc).__name__}).") from exc
    if r.status_code != 200:
        raise HTTPException(status_code=502 if r.status_code >= 500 else r.status_code,
                            detail=f"Inference host returned HTTP {r.status_code}.")
    return r.json()


HOST_CALLS = {"n": 0}  # real depth-host calls made by this process (logged; baked library depth never counts)
MODEL_HOST = "Cloud Run model service"  # how the UI names DW2_MODEL_URL (never the internal URL)


def model_url() -> str:
    return os.environ.get("DW2_MODEL_URL", "").strip().rstrip("/")


_id_tokens: dict[str, tuple[str, float]] = {}


def _id_token(audience: str) -> str:
    """An ID token for this service's own service account (Cloud Run metadata server), reused for 50 minutes.
    Local dev only: DW2_MODEL_GCLOUD_TOKEN=1 falls back to `gcloud auth print-identity-token` (needs run.invoker)."""
    tok, exp = _id_tokens.get(audience, ("", 0.0))
    if time.time() < exp:
        return tok
    import google.auth.transport.requests
    import google.oauth2.id_token
    try:
        tok = google.oauth2.id_token.fetch_id_token(google.auth.transport.requests.Request(), audience)
    except Exception:  # noqa: BLE001 - local dev: no metadata server; use the developer's own gcloud identity
        if not os.environ.get("DW2_MODEL_GCLOUD_TOKEN"):
            raise
        import subprocess
        tok = subprocess.run(["gcloud", "auth", "print-identity-token"], capture_output=True, text=True,
                             check=True, timeout=30).stdout.strip()
    _id_tokens[audience] = (tok, time.time() + 50 * 60)
    return tok


async def call_model(path: str, filename: str, body: bytes, content_type: str, data: dict | None = None) -> dict:
    """POST to the model service with an ID token. Raises HTTPException (502 when there is no token or no host)."""
    url = model_url()
    try:
        token = await asyncio.to_thread(_id_token, url)
    except Exception as exc:  # noqa: BLE001 - no metadata server (local dev) or no credentials
        raise HTTPException(status_code=502, detail=f"No ID token for the model service ({type(exc).__name__}).") from exc
    return await _call_http(url, filename, body, content_type, {"Authorization": f"Bearer {token}"}, path, data)


async def _forward(filename: str, body: bytes, content_type: str) -> dict:
    log = logging.getLogger("uvicorn.error")
    HOST_CALLS["n"] += 1
    fallback = None
    if model_url():
        log.info("depth host call #%d -> %s (%s)", HOST_CALLS["n"], MODEL_HOST, filename)
        try:
            return {**await call_model("/predict", filename, body, content_type), "host": MODEL_HOST}
        except HTTPException as exc:
            if exc.status_code < 500:
                raise
            fallback = f"{MODEL_HOST} failed ({exc.detail}); the Hugging Face Space answered instead."
            log.warning("depth host call #%d: %s", HOST_CALLS["n"], fallback)
    url = host()
    sid = space_id(url)
    log.info("depth host call #%d -> %s (%s)", HOST_CALLS["n"], sid or url, filename)
    out = await (_call_space(sid, filename, body) if sid else _call_http(url, filename, body, content_type))
    return {**out, "host": sid or url, **({"fallback": fallback} if fallback else {})}


# ------------------------------------------------------------------ routes
@router.get("/status")
def status() -> dict:
    url = host()
    sid = space_id(url)
    if sid:
        try:
            from huggingface_hub import HfApi
            rt = HfApi(token=hf_token()).get_space_runtime(sid).raw
            out = {"kind": "space", "host": sid, "reachable": rt.get("stage") == "RUNNING",
                   "stage": rt.get("stage"), "hardware": (rt.get("hardware") or {}).get("current"),
                   "token_configured": bool(hf_token())}
        except Exception as exc:  # noqa: BLE001
            out = {"kind": "space", "host": sid, "reachable": False, "error": type(exc).__name__}
    else:
        try:
            r = httpx.get(f"{url}/health", timeout=15)
            r.raise_for_status()
            out = {"kind": "http", "host": url, "reachable": True, **r.json()}
        except httpx.HTTPError as exc:
            out = {"kind": "http", "host": url, "reachable": False, "error": type(exc).__name__}
    return {"hosts": [out], "space_request_auth": auth_audit()}


@router.post("/relative")
async def relative(file: UploadFile = File(...)) -> dict:
    return await _forward(file.filename or "image.jpg", await file.read(), file.content_type or "application/octet-stream")


async def _preview_bytes(source: str, item_id: str) -> bytes:
    """The same 1024 px preview the UI shows for a library item or an input (upload / CDSE scene)."""
    if source == "input":
        try:
            return store.preview_path(item_id, "image").read_bytes()
        except store.InputError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    item = catalog.get(item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Unknown library item.")
    if library_store.mode() == "local":
        return catalog.image_path(item, "preview").read_bytes()
    if library_store.mode() == "bundle":
        path = library_store.local_asset(item, "preview")
        if path is None:
            raise HTTPException(status_code=409, detail="Download this tile first.")
        return path.read_bytes()
    if not library_store.is_public(item):
        return library_store.private_file(item, "preview").read_bytes()
    names = item.get("assets") or library_store.asset_names(item)
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        r = await client.get(library_store.release_url(names["preview"]))
    r.raise_for_status()
    return r.content


@router.post("/relative/{source}/{item_id}")
async def relative_for(source: str, item_id: str) -> dict:
    """Relative depth for a selected library item or input, by id (the UI's generation step)."""
    if source not in ("library", "input"):
        raise HTTPException(status_code=404, detail="source must be 'library' or 'input'.")
    try:
        body = await _preview_bytes(source, item_id)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"Could not fetch the preview ({type(exc).__name__}).") from exc
    if source == "library":
        from backend.generation import pipeline
        baked = pipeline.baked_depth(catalog.get(item_id), body)
        if baked is not None:
            return baked
    return await _forward(f"{item_id}.jpg", body, "image/jpeg")
