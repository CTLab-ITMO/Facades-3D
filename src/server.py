from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import signal
import tempfile
import threading
import traceback
from contextlib import suppress
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from types import FrameType
from typing import Awaitable, Callable, Literal

import uvicorn
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field, ValidationError
from starlette.concurrency import run_in_threadpool

from default_configs import DEFAULT_GENERATION_CONFIG, GenerationParametersConfig
from gen_facades import generate_facade_scene
from gen_wall import release_generation_memory
from request_control import (
    CancellationOrigin,
    GenerationCancelledError,
    RequestCancellation,
    raise_if_cancelled,
)

app = FastAPI(
    title="3D Building Facades Generation Server",
    description="Generate detailed GLB building exteriors from OBJ mass models.",
)

logger = logging.getLogger("uvicorn.error")

HTTP_DISCONNECT_POLL_INTERVAL_SECONDS = 0.25
WEBSOCKET_BINARY_CHUNK_SIZE = 4 * 1024 * 1024

StatusCallback = Callable[..., Awaitable[None]]
ProgressCallback = Callable[[int, int], None]

_CLIENT_SIDE_CANCELLATION_ORIGINS: set[CancellationOrigin] = {
    "client_cancel",
    "connection_drop",
}


def _log_cancelled_request(
    transport: str,
    cancellation: RequestCancellation,
) -> None:
    message = "%s generation request terminated: %s"
    args = (transport, cancellation.reason)

    if cancellation.origin == "server_signal":
        logger.warning(message, *args)
    elif cancellation.origin in _CLIENT_SIDE_CANCELLATION_ORIGINS:
        logger.info(message, *args)
    else:
        logger.warning(message, *args)


class GenerationFailure(RuntimeError):
    def __init__(
        self,
        original_type: type[Exception],
        message: str,
        traceback_text: str,
    ) -> None:
        super().__init__(message)
        self.original_type = original_type
        self.traceback_text = traceback_text


class GenerationCoordinator:
    def __init__(self) -> None:
        self._generation_lock = asyncio.Lock()
        self._state_lock = threading.Lock()
        self._queued_cancellations: list[RequestCancellation] = []
        self._active_cancellation: RequestCancellation | None = None

    def has_work(self) -> bool:
        with self._state_lock:
            return self._active_cancellation is not None or bool(self._queued_cancellations)

    def cancel_all(self, reason: str) -> tuple[bool, int]:
        with self._state_lock:
            active_cancellation = self._active_cancellation
            queued_cancellations = tuple(self._queued_cancellations)

        if active_cancellation is not None:
            active_cancellation.cancel(reason, origin="server_signal")
        for cancellation in queued_cancellations:
            cancellation.cancel(reason, origin="server_signal")

        return active_cancellation is not None, len(queued_cancellations)

    async def run(
        self,
        function: Callable[..., bytes],
        *args: object,
        cancellation: RequestCancellation,
        on_queued: Callable[[int], Awaitable[None]] | None = None,
        on_started: Callable[[], Awaitable[None]] | None = None,
    ) -> bytes:
        loop = asyncio.get_running_loop()
        cancellation.bind_loop(loop)

        with self._state_lock:
            self._queued_cancellations.append(cancellation)
            queue_position = len(self._queued_cancellations)
            if self._active_cancellation is not None:
                queue_position += 1

        if on_queued is not None:
            await on_queued(queue_position)

        acquire_task = asyncio.create_task(self._generation_lock.acquire())
        cancellation_task = asyncio.create_task(cancellation.wait())
        lock_acquired = False

        try:
            completed, _ = await asyncio.wait(
                {acquire_task, cancellation_task},
                return_when=asyncio.FIRST_COMPLETED,
            )

            if acquire_task in completed:
                await acquire_task
                lock_acquired = True

            if cancellation.is_cancelled():
                if not lock_acquired:
                    acquire_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await acquire_task
                raise GenerationCancelledError(cancellation.reason)

            if not lock_acquired:
                await acquire_task
                lock_acquired = True
        except BaseException:
            if not lock_acquired:
                acquire_task.cancel()
                with suppress(asyncio.CancelledError):
                    await acquire_task
            with self._state_lock:
                with suppress(ValueError):
                    self._queued_cancellations.remove(cancellation)
            if lock_acquired:
                self._generation_lock.release()
            raise
        finally:
            cancellation_task.cancel()
            with suppress(asyncio.CancelledError):
                await cancellation_task

        with self._state_lock:
            with suppress(ValueError):
                self._queued_cancellations.remove(cancellation)
            if cancellation.is_cancelled():
                self._generation_lock.release()
                raise GenerationCancelledError(cancellation.reason)
            self._active_cancellation = cancellation

        result: bytes | None = None
        failure: GenerationFailure | None = None
        handler_cancelled: asyncio.CancelledError | None = None
        worker_task: asyncio.Task[bytes] | None = None

        try:
            if on_started is not None:
                await on_started()

            worker_task = asyncio.create_task(
                run_in_threadpool(function, *args, cancellation.thread_event)
            )
            try:
                result = await asyncio.shield(worker_task)
            except asyncio.CancelledError as error:
                handler_cancelled = error
                cancellation.cancel("Request handler was cancelled", origin="handler")
                with suppress(BaseException):
                    await asyncio.shield(worker_task)
            except Exception as error:
                failure = GenerationFailure(
                    original_type=type(error),
                    message=str(error),
                    traceback_text=traceback.format_exc(),
                )
                error.__traceback__ = None
        except asyncio.CancelledError as error:
            handler_cancelled = error
            cancellation.cancel("Request handler was cancelled", origin="handler")
        except Exception as error:
            failure = GenerationFailure(
                original_type=type(error),
                message=str(error),
                traceback_text=traceback.format_exc(),
            )
            error.__traceback__ = None
        finally:
            if worker_task is not None and not worker_task.done():
                cancellation.cancel("Request handler was cancelled", origin="handler")
                with suppress(BaseException):
                    await asyncio.shield(worker_task)

            try:
                await run_in_threadpool(release_generation_memory)
            except Exception as cleanup_error:
                cleanup_traceback = traceback.format_exc()
                cleanup_error.__traceback__ = None
                if failure is None:
                    failure = GenerationFailure(
                        original_type=type(cleanup_error),
                        message=f"CUDA cleanup failed: {cleanup_error}",
                        traceback_text=cleanup_traceback,
                    )
                else:
                    failure.traceback_text += f"\nCUDA cleanup also failed:\n{cleanup_traceback}"
            finally:
                with self._state_lock:
                    self._active_cancellation = None
                self._generation_lock.release()

        if handler_cancelled is not None:
            raise handler_cancelled
        if failure is not None:
            raise failure
        if result is None:
            raise RuntimeError("Generation completed without a result")
        return result


_generation_coordinator = GenerationCoordinator()


class CancellationAwareServer(uvicorn.Server):
    _CANCELLATION_SIGNALS = {signal.SIGINT, signal.SIGTERM}

    def handle_exit(self, sig: int, frame: FrameType | None) -> None:
        if sig in self._CANCELLATION_SIGNALS and _generation_coordinator.has_work():
            signal_name = signal.Signals(sig).name
            active_cancelled, queued_cancelled = _generation_coordinator.cancel_all(
                f"Generation was cancelled by {signal_name}"
            )
            logger.warning(
                "%s received while generation work exists: cancellation requested "
                "for active=%s and queued=%d request(s); server will keep running",
                signal_name,
                active_cancelled,
                queued_cancelled,
            )
            return

        super().handle_exit(sig, frame)


class WebSocketFileDescription(BaseModel):
    filename: str
    size: int = Field(gt=0)


class WebSocketGenerationParameters(BaseModel):
    pixels_per_meter: int = Field(gt=0)
    cluster_count: int = Field(gt=0)
    output_filename: str = DEFAULT_GENERATION_CONFIG.output_filename
    seed: int | None = DEFAULT_GENERATION_CONFIG.seed
    temp_root: Path | None = DEFAULT_GENERATION_CONFIG.temp_root

    prompt: str = DEFAULT_GENERATION_CONFIG.prompt
    negative_prompt: str = DEFAULT_GENERATION_CONFIG.negative_prompt
    diffusion_steps: int = DEFAULT_GENERATION_CONFIG.diffusion_steps
    guidance_scale: float = DEFAULT_GENERATION_CONFIG.guidance_scale
    controlnet_conditioning_scale: float = DEFAULT_GENERATION_CONFIG.controlnet_conditioning_scale
    control_guidance_start: float = DEFAULT_GENERATION_CONFIG.control_guidance_start
    control_guidance_end: float = DEFAULT_GENERATION_CONFIG.control_guidance_end
    cross_attention_scale: float = DEFAULT_GENERATION_CONFIG.cross_attention_scale
    style_ref_scale: float = DEFAULT_GENERATION_CONFIG.style_ref_scale

    sparse_structure_steps: int = DEFAULT_GENERATION_CONFIG.sparse_structure_steps
    sparse_structure_cfg_strength: float = DEFAULT_GENERATION_CONFIG.sparse_structure_cfg_strength
    slat_steps: int = DEFAULT_GENERATION_CONFIG.slat_steps
    slat_cfg_strength: float = DEFAULT_GENERATION_CONFIG.slat_cfg_strength
    trellis_mode: str = DEFAULT_GENERATION_CONFIG.trellis_mode

    mesh_simplify: float = DEFAULT_GENERATION_CONFIG.mesh_simplify
    texture_size: int = DEFAULT_GENERATION_CONFIG.texture_size
    border_size: int = DEFAULT_GENERATION_CONFIG.border_size
    texture_brightness_factor: float = DEFAULT_GENERATION_CONFIG.texture_brightness_factor
    texture_postprocess_shrink_px: int = DEFAULT_GENERATION_CONFIG.texture_postprocess_shrink_px
    depth_scale_reference: float = DEFAULT_GENERATION_CONFIG.depth_scale_reference
    max_wall_aspect_ratio: float = DEFAULT_GENERATION_CONFIG.max_wall_aspect_ratio

    def to_generation_config(self) -> GenerationParametersConfig:
        return replace(
            DEFAULT_GENERATION_CONFIG,
            output_filename=self.output_filename,
            cluster_count=self.cluster_count,
            seed=self.seed,
            temp_root=self.temp_root,
            prompt=self.prompt,
            negative_prompt=self.negative_prompt,
            diffusion_steps=self.diffusion_steps,
            guidance_scale=self.guidance_scale,
            controlnet_conditioning_scale=self.controlnet_conditioning_scale,
            control_guidance_start=self.control_guidance_start,
            control_guidance_end=self.control_guidance_end,
            cross_attention_scale=self.cross_attention_scale,
            style_ref_scale=self.style_ref_scale,
            sparse_structure_steps=self.sparse_structure_steps,
            sparse_structure_cfg_strength=self.sparse_structure_cfg_strength,
            slat_steps=self.slat_steps,
            slat_cfg_strength=self.slat_cfg_strength,
            trellis_mode=self.trellis_mode,
            mesh_simplify=self.mesh_simplify,
            texture_size=self.texture_size,
            border_size=self.border_size,
            texture_brightness_factor=self.texture_brightness_factor,
            texture_postprocess_shrink_px=self.texture_postprocess_shrink_px,
            depth_scale_reference=self.depth_scale_reference,
            max_wall_aspect_ratio=self.max_wall_aspect_ratio,
        )


class WebSocketGenerationStart(BaseModel):
    type: Literal["generate"]
    parameters: WebSocketGenerationParameters
    input_model: WebSocketFileDescription
    style_reference: WebSocketFileDescription | None = None


def _decode_style_reference(image_bytes: bytes | None) -> Image.Image | None:
    if image_bytes is None:
        return None

    with Image.open(BytesIO(image_bytes)) as image:
        return image.convert("RGB").copy()


def _normalize_output_filename(filename: str) -> str:
    safe_name = Path(filename).name
    if not safe_name:
        safe_name = DEFAULT_GENERATION_CONFIG.output_filename
    if Path(safe_name).suffix.lower() != ".glb":
        safe_name = f"{safe_name}.glb"
    return safe_name


def create_generation_config(
    *,
    output_filename: str,
    cluster_count: int,
    seed: int | None,
    temp_root: Path | None,
    style_ref_scale: float,
    max_wall_aspect_ratio: float,
    prompt: str,
    negative_prompt: str,
    diffusion_steps: int,
    guidance_scale: float,
    controlnet_conditioning_scale: float,
    control_guidance_start: float,
    control_guidance_end: float,
    cross_attention_scale: float,
    sparse_structure_steps: int,
    sparse_structure_cfg_strength: float,
    slat_steps: int,
    slat_cfg_strength: float,
    trellis_mode: str,
    mesh_simplify: float,
    texture_size: int,
    border_size: int,
    texture_brightness_factor: float,
    texture_postprocess_shrink_px: int,
    depth_scale_reference: float,
) -> GenerationParametersConfig:
    return replace(
        DEFAULT_GENERATION_CONFIG,
        output_filename=output_filename,
        cluster_count=cluster_count,
        seed=seed,
        temp_root=temp_root,
        style_ref_scale=style_ref_scale,
        max_wall_aspect_ratio=max_wall_aspect_ratio,
        prompt=prompt,
        negative_prompt=negative_prompt,
        diffusion_steps=diffusion_steps,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        control_guidance_start=control_guidance_start,
        control_guidance_end=control_guidance_end,
        cross_attention_scale=cross_attention_scale,
        sparse_structure_steps=sparse_structure_steps,
        sparse_structure_cfg_strength=sparse_structure_cfg_strength,
        slat_steps=slat_steps,
        slat_cfg_strength=slat_cfg_strength,
        trellis_mode=trellis_mode,
        mesh_simplify=mesh_simplify,
        texture_size=texture_size,
        border_size=border_size,
        texture_brightness_factor=texture_brightness_factor,
        texture_postprocess_shrink_px=texture_postprocess_shrink_px,
        depth_scale_reference=depth_scale_reference,
    )


def _generate_glb(
    obj_bytes: bytes,
    style_reference_bytes: bytes | None,
    pixels_per_meter: int,
    config: GenerationParametersConfig,
    progress_callback: ProgressCallback | None,
    cancel_event: threading.Event,
) -> bytes:
    raise_if_cancelled(cancel_event)
    style_reference = _decode_style_reference(style_reference_bytes)

    with tempfile.TemporaryDirectory(prefix="facade-server-", dir=config.temp_root) as temporary_directory:
        root = Path(temporary_directory)
        input_path = root / "input.obj"
        visual_path = root / "visual"
        input_path.write_bytes(obj_bytes)

        raise_if_cancelled(cancel_event)
        scene, _, _ = generate_facade_scene(
            input_path=input_path,
            visual_path=visual_path,
            pixels_per_meter=pixels_per_meter,
            style_ref=style_reference,
            config=config,
            cancel_event=cancel_event,
            progress_callback=progress_callback,
        )
        raise_if_cancelled(cancel_event)
        result = scene.export(file_type="glb")
        raise_if_cancelled(cancel_event)

    if not isinstance(result, bytes):
        raise RuntimeError("GLB exporter returned an unexpected result type")
    return result


def _raise_request_error(status_code: int, detail: str) -> None:
    raise HTTPException(status_code=status_code, detail=detail)


async def _monitor_http_disconnect(
    request: Request,
    cancellation: RequestCancellation,
) -> None:
    while not cancellation.is_cancelled():
        if await request.is_disconnected():
            cancellation.cancel("HTTP client disconnected", origin="connection_drop")
            return
        await asyncio.sleep(HTTP_DISCONNECT_POLL_INTERVAL_SECONDS)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/generate", response_class=Response)
async def generate(
    request: Request,
    input_model: UploadFile = File(..., description="Input mass model in OBJ format."),
    style_reference: UploadFile | None = File(
        None,
        description="Optional style-reference image. Omit it to disable IP-Adapter conditioning.",
    ),
    output_filename: str = Form(DEFAULT_GENERATION_CONFIG.output_filename),
    pixels_per_meter: int = Form(...),
    style_ref_scale: float = Form(DEFAULT_GENERATION_CONFIG.style_ref_scale),
    cluster_count: int = Form(...),
    max_wall_aspect_ratio: float = Form(DEFAULT_GENERATION_CONFIG.max_wall_aspect_ratio),
    seed: int | None = Form(DEFAULT_GENERATION_CONFIG.seed),
    temp_root: Path | None = Form(DEFAULT_GENERATION_CONFIG.temp_root),
    prompt: str = Form(DEFAULT_GENERATION_CONFIG.prompt),
    negative_prompt: str = Form(DEFAULT_GENERATION_CONFIG.negative_prompt),
    diffusion_steps: int = Form(DEFAULT_GENERATION_CONFIG.diffusion_steps),
    guidance_scale: float = Form(DEFAULT_GENERATION_CONFIG.guidance_scale),
    controlnet_conditioning_scale: float = Form(DEFAULT_GENERATION_CONFIG.controlnet_conditioning_scale),
    control_guidance_start: float = Form(DEFAULT_GENERATION_CONFIG.control_guidance_start),
    control_guidance_end: float = Form(DEFAULT_GENERATION_CONFIG.control_guidance_end),
    cross_attention_scale: float = Form(DEFAULT_GENERATION_CONFIG.cross_attention_scale),
    sparse_structure_steps: int = Form(DEFAULT_GENERATION_CONFIG.sparse_structure_steps),
    sparse_structure_cfg_strength: float = Form(DEFAULT_GENERATION_CONFIG.sparse_structure_cfg_strength),
    slat_steps: int = Form(DEFAULT_GENERATION_CONFIG.slat_steps),
    slat_cfg_strength: float = Form(DEFAULT_GENERATION_CONFIG.slat_cfg_strength),
    trellis_mode: str = Form(DEFAULT_GENERATION_CONFIG.trellis_mode),
    mesh_simplify: float = Form(DEFAULT_GENERATION_CONFIG.mesh_simplify),
    texture_size: int = Form(DEFAULT_GENERATION_CONFIG.texture_size),
    border_size: int = Form(DEFAULT_GENERATION_CONFIG.border_size),
    texture_brightness_factor: float = Form(DEFAULT_GENERATION_CONFIG.texture_brightness_factor),
    texture_postprocess_shrink_px: int = Form(DEFAULT_GENERATION_CONFIG.texture_postprocess_shrink_px),
    depth_scale_reference: float = Form(DEFAULT_GENERATION_CONFIG.depth_scale_reference),
) -> Response:
    obj_bytes = await input_model.read()
    if not obj_bytes:
        _raise_request_error(400, "The uploaded OBJ file is empty")

    style_reference_bytes = None
    if style_reference is not None:
        style_reference_bytes = await style_reference.read()
        if not style_reference_bytes:
            _raise_request_error(400, "The uploaded style-reference image is empty")

    cancellation = RequestCancellation()
    disconnect_monitor = asyncio.create_task(
        _monitor_http_disconnect(request, cancellation)
    )

    try:
        config = create_generation_config(
            output_filename=output_filename,
            cluster_count=cluster_count,
            seed=seed,
            temp_root=temp_root,
            style_ref_scale=style_ref_scale,
            max_wall_aspect_ratio=max_wall_aspect_ratio,
            prompt=prompt,
            negative_prompt=negative_prompt,
            diffusion_steps=diffusion_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            control_guidance_start=control_guidance_start,
            control_guidance_end=control_guidance_end,
            cross_attention_scale=cross_attention_scale,
            sparse_structure_steps=sparse_structure_steps,
            sparse_structure_cfg_strength=sparse_structure_cfg_strength,
            slat_steps=slat_steps,
            slat_cfg_strength=slat_cfg_strength,
            trellis_mode=trellis_mode,
            mesh_simplify=mesh_simplify,
            texture_size=texture_size,
            border_size=border_size,
            texture_brightness_factor=texture_brightness_factor,
            texture_postprocess_shrink_px=texture_postprocess_shrink_px,
            depth_scale_reference=depth_scale_reference,
        )

        glb_bytes = await _generation_coordinator.run(
            _generate_glb,
            obj_bytes,
            style_reference_bytes,
            pixels_per_meter,
            config,
            None,
            cancellation=cancellation,
        )
    except GenerationCancelledError as error:
        _log_cancelled_request("HTTP", cancellation)
        _raise_request_error(503, cancellation.reason)
    except GenerationFailure as error:
        if issubclass(error.original_type, GenerationCancelledError):
            _log_cancelled_request("HTTP", cancellation)
            _raise_request_error(503, cancellation.reason)

        if issubclass(error.original_type, (ValueError, UnidentifiedImageError)):
            logger.warning("Generation request rejected: %s", error)
            _raise_request_error(400, str(error))

        logger.error(
            "Unhandled error while processing POST /generate\n%s",
            error.traceback_text,
        )
        _raise_request_error(500, f"Generation failed: {error}")
    except (ValueError, UnidentifiedImageError) as error:
        logger.warning("Generation request rejected: %s", error)
        _raise_request_error(400, str(error))
    except asyncio.CancelledError:
        cancellation.cancel("HTTP request handler was cancelled", origin="handler")
        raise
    except Exception:
        logger.exception("Unhandled error while processing POST /generate")
        _raise_request_error(500, "Generation failed due to an internal server error")
    finally:
        disconnect_monitor.cancel()
        with suppress(asyncio.CancelledError):
            await disconnect_monitor

    filename = _normalize_output_filename(config.output_filename)
    return Response(
        content=glb_bytes,
        media_type="model/gltf-binary",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.websocket("/ws/health")
async def websocket_health(websocket: WebSocket) -> None:
    await websocket.accept()
    await websocket.send_json({"type": "health", "status": "ok"})
    await websocket.close(code=1000)


async def _receive_websocket_file(
    websocket: WebSocket,
    file_description: WebSocketFileDescription,
    cancellation: RequestCancellation,
) -> bytes:
    data = bytearray()

    while len(data) < file_description.size:
        message = await websocket.receive()
        message_type = message.get("type")

        if message_type == "websocket.disconnect":
            cancellation.cancel("WebSocket client disconnected", origin="connection_drop")
            raise WebSocketDisconnect(message.get("code", 1000))

        binary_data = message.get("bytes")
        if binary_data is not None:
            data.extend(binary_data)
            if len(data) > file_description.size:
                raise ValueError(
                    f"Received more bytes than declared for {file_description.filename}"
                )
            continue

        text_data = message.get("text")
        if text_data is not None:
            try:
                control_message = json.loads(text_data)
            except json.JSONDecodeError as error:
                raise ValueError("Expected binary file data or a cancel message") from error

            if control_message.get("type") == "cancel":
                cancellation.cancel(
                    "Generation was cancelled by the WebSocket client",
                    origin="client_cancel",
                )
                raise GenerationCancelledError(cancellation.reason)

            raise ValueError("Only a cancel message is allowed during file upload")

    return bytes(data)


async def _send_websocket_cancellation(
    websocket: WebSocket,
    reason: str,
) -> None:
    with suppress(WebSocketDisconnect, RuntimeError):
        await websocket.send_json({"type": "cancelled", "reason": reason})
    with suppress(WebSocketDisconnect, RuntimeError):
        await websocket.close(code=1000)


async def _send_websocket_error(
    websocket: WebSocket,
    status_code: int,
    detail: str,
) -> None:
    with suppress(WebSocketDisconnect, RuntimeError):
        await websocket.send_json(
            {"type": "error", "status_code": status_code, "detail": detail}
        )
    with suppress(WebSocketDisconnect, RuntimeError):
        await websocket.close(code=1011 if status_code >= 500 else 1008)


async def _handle_websocket_generation(
    websocket: WebSocket,
    start_message: WebSocketGenerationStart,
) -> None:
    cancellation = RequestCancellation()
    cancellation.bind_loop(asyncio.get_running_loop())

    try:
        await websocket.send_json(
            {
                "type": "upload_ready",
                "chunk_size": WEBSOCKET_BINARY_CHUNK_SIZE,
                "files": [
                    start_message.input_model.filename,
                    *(
                        [start_message.style_reference.filename]
                        if start_message.style_reference is not None
                        else []
                    ),
                ],
            }
        )

        obj_bytes = await _receive_websocket_file(
            websocket,
            start_message.input_model,
            cancellation,
        )
        style_reference_bytes = None
        if start_message.style_reference is not None:
            style_reference_bytes = await _receive_websocket_file(
                websocket,
                start_message.style_reference,
                cancellation,
            )

        await websocket.send_json({"type": "upload_received"})
    except GenerationCancelledError:
        _log_cancelled_request("WebSocket", cancellation)
        await _send_websocket_cancellation(websocket, cancellation.reason)
        return
    except WebSocketDisconnect:
        _log_cancelled_request("WebSocket", cancellation)
        return
    except Exception as error:
        await _send_websocket_error(websocket, 400, str(error))
        return

    parameters = start_message.parameters
    config = parameters.to_generation_config()
    progress_queue: asyncio.Queue[dict[str, object]] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def report_progress(completed: int, total: int) -> None:
        percentage = 100.0 if total == 0 else completed * 100.0 / total
        message: dict[str, object] = {
            "type": "progress",
            "stage": "generating_wall_models",
            "completed": completed,
            "total": total,
            "percent": round(percentage, 2),
        }
        with suppress(RuntimeError):
            loop.call_soon_threadsafe(progress_queue.put_nowait, message)

    async def on_queued(position: int) -> None:
        await progress_queue.put({"type": "queued", "position": position})

    async def on_started() -> None:
        await progress_queue.put({"type": "started"})

    generation_task = asyncio.create_task(
        _generation_coordinator.run(
            _generate_glb,
            obj_bytes,
            style_reference_bytes,
            parameters.pixels_per_meter,
            config,
            report_progress,
            cancellation=cancellation,
            on_queued=on_queued,
            on_started=on_started,
        )
    )
    receive_task: asyncio.Task[dict[str, object]] | None = asyncio.create_task(
        websocket.receive()
    )
    progress_task: asyncio.Task[dict[str, object]] | None = asyncio.create_task(
        progress_queue.get()
    )
    cancellation_task: asyncio.Task[None] | None = asyncio.create_task(
        cancellation.wait()
    )
    client_connected = True
    cancellation_acknowledged = False

    try:
        while not generation_task.done():
            wait_tasks: set[asyncio.Task[object]] = {generation_task}
            if receive_task is not None:
                wait_tasks.add(receive_task)
            if progress_task is not None:
                wait_tasks.add(progress_task)
            if cancellation_task is not None:
                wait_tasks.add(cancellation_task)

            completed_tasks, _ = await asyncio.wait(
                wait_tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )

            if progress_task is not None and progress_task in completed_tasks:
                progress_message = progress_task.result()
                try:
                    await websocket.send_json(progress_message)
                except (WebSocketDisconnect, RuntimeError):
                    client_connected = False
                    cancellation.cancel("WebSocket client disconnected", origin="connection_drop")
                progress_queue.task_done()
                progress_task = (
                    asyncio.create_task(progress_queue.get())
                    if client_connected and not generation_task.done()
                    else None
                )

            if receive_task is not None and receive_task in completed_tasks:
                message = receive_task.result()
                if message.get("type") == "websocket.disconnect":
                    client_connected = False
                    cancellation.cancel("WebSocket client disconnected", origin="connection_drop")
                    receive_task = None
                else:
                    text_data = message.get("text")
                    if text_data is None:
                        await websocket.send_json(
                            {
                                "type": "error",
                                "status_code": 400,
                                "detail": "Only JSON control messages are accepted during generation",
                            }
                        )
                    else:
                        try:
                            control_message = json.loads(text_data)
                        except json.JSONDecodeError:
                            await websocket.send_json(
                                {
                                    "type": "error",
                                    "status_code": 400,
                                    "detail": "Invalid JSON control message",
                                }
                            )
                        else:
                            if control_message.get("type") == "cancel":
                                cancellation.cancel(
                                    "Generation was cancelled by the WebSocket client",
                                    origin="client_cancel",
                                )
                            else:
                                await websocket.send_json(
                                    {
                                        "type": "error",
                                        "status_code": 400,
                                        "detail": "Unknown control message",
                                    }
                                )
                    receive_task = (
                        asyncio.create_task(websocket.receive())
                        if client_connected and not cancellation.is_cancelled()
                        else None
                    )

            if cancellation_task is not None and cancellation_task in completed_tasks:
                if client_connected and not cancellation_acknowledged:
                    try:
                        await websocket.send_json(
                            {
                                "type": "cancel_requested",
                                "reason": cancellation.reason,
                            }
                        )
                        cancellation_acknowledged = True
                    except (WebSocketDisconnect, RuntimeError):
                        client_connected = False
                cancellation_task = None

        for task in (receive_task, progress_task, cancellation_task):
            if task is not None:
                task.cancel()
                with suppress(asyncio.CancelledError, WebSocketDisconnect):
                    await task

        if client_connected:
            while not progress_queue.empty():
                progress_message = progress_queue.get_nowait()
                await websocket.send_json(progress_message)
                progress_queue.task_done()

        try:
            glb_bytes = generation_task.result()
        except GenerationCancelledError:
            _log_cancelled_request("WebSocket", cancellation)
            if client_connected:
                await _send_websocket_cancellation(websocket, cancellation.reason)
            return
        except GenerationFailure as error:
            if issubclass(error.original_type, GenerationCancelledError):
                _log_cancelled_request("WebSocket", cancellation)
                if client_connected:
                    await _send_websocket_cancellation(websocket, cancellation.reason)
                return

            if issubclass(error.original_type, (ValueError, UnidentifiedImageError)):
                if client_connected:
                    await _send_websocket_error(websocket, 400, str(error))
                return

            logger.error(
                "Unhandled error while processing WebSocket generation\n%s",
                error.traceback_text,
            )
            if client_connected:
                await _send_websocket_error(websocket, 500, f"Generation failed: {error}")
            return
        except Exception as error:
            logger.exception("Unhandled WebSocket generation error")
            if client_connected:
                await _send_websocket_error(websocket, 500, str(error))
            return

        if cancellation.is_cancelled():
            _log_cancelled_request("WebSocket", cancellation)
            if client_connected:
                await _send_websocket_cancellation(websocket, cancellation.reason)
            return

        if not client_connected:
            return

        filename = _normalize_output_filename(config.output_filename)
        chunk_count = math.ceil(len(glb_bytes) / WEBSOCKET_BINARY_CHUNK_SIZE)
        await websocket.send_json(
            {
                "type": "result_ready",
                "filename": filename,
                "content_type": "model/gltf-binary",
                "size": len(glb_bytes),
                "chunk_size": WEBSOCKET_BINARY_CHUNK_SIZE,
                "chunk_count": chunk_count,
            }
        )

        for offset in range(0, len(glb_bytes), WEBSOCKET_BINARY_CHUNK_SIZE):
            await websocket.send_bytes(
                glb_bytes[offset : offset + WEBSOCKET_BINARY_CHUNK_SIZE]
            )

        await websocket.send_json({"type": "completed"})
        await websocket.close(code=1000)
    except WebSocketDisconnect:
        cancellation.cancel(
            "WebSocket client disconnected",
            origin="connection_drop",
        )
        with suppress(BaseException):
            await generation_task
        _log_cancelled_request("WebSocket", cancellation)
    except asyncio.CancelledError:
        cancellation.cancel(
            "WebSocket request handler was cancelled",
            origin="handler",
        )
        with suppress(BaseException):
            await generation_task
        _log_cancelled_request("WebSocket", cancellation)
        raise
    finally:
        for task in (receive_task, progress_task, cancellation_task):
            if task is not None and not task.done():
                task.cancel()
        if not generation_task.done():
            cancellation.cancel(
                "WebSocket request handler was terminated",
                origin="handler",
            )
            with suppress(BaseException):
                await generation_task


@app.websocket("/ws/generate")
async def websocket_generate(websocket: WebSocket) -> None:
    await websocket.accept()

    try:
        raw_message = await websocket.receive_json()
        start_message = WebSocketGenerationStart(**raw_message)
    except WebSocketDisconnect:
        logger.info("WebSocket generation client disconnected before request start")
        return
    except ValidationError as error:
        await _send_websocket_error(websocket, 400, str(error))
        return
    except Exception as error:
        await _send_websocket_error(websocket, 400, str(error))
        return

    await _handle_websocket_generation(websocket, start_message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a detailed GLB 3D model of one or more buildings from an OBJ mass model."
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Server host address (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not 1 <= args.port <= 65535:
        raise SystemExit("Port must be between 1 and 65535.")

    print(f"Starting server at http://{args.host}:{args.port}", flush=True)

    server_config = uvicorn.Config(
        app,
        host=args.host,
        port=args.port,
    )
    CancellationAwareServer(server_config).run()
