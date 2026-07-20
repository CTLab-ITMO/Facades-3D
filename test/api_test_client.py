from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
from contextlib import suppress
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse

try:
    import httpx
except ImportError as error:  # pragma: no cover - convenience for manual use
    raise SystemExit("Install test dependencies with: pip install httpx websockets") from error

try:
    import websockets
except ImportError as error:  # pragma: no cover - convenience for manual use
    raise SystemExit("Install test dependencies with: pip install httpx websockets") from error


CLIENT_CHUNK_SIZE = 4 * 1024 * 1024


def _websocket_base_url(http_base_url: str) -> str:
    parsed = urlparse(http_base_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return urlunparse((scheme, parsed.netloc, parsed.path.rstrip("/"), "", "", ""))


def _load_parameter_overrides(value: str | None) -> dict[str, Any]:
    if value is None:
        return {}

    possible_path = Path(value)
    if possible_path.exists():
        return json.loads(possible_path.read_text(encoding="utf-8"))
    return json.loads(value)


def _generation_parameters(args: argparse.Namespace) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        "pixels_per_meter": args.pixels_per_meter,
        "cluster_count": args.cluster_count,
        "output_filename": args.remote_output_filename,
    }
    parameters.update(_load_parameter_overrides(args.parameters_json))
    return parameters


def _rest_form_data(parameters: dict[str, Any]) -> dict[str, str]:
    return {
        key: str(value)
        for key, value in parameters.items()
        if value is not None
    }


def _rest_files(args: argparse.Namespace) -> dict[str, tuple[str, bytes, str]]:
    if args.obj is None:
        raise ValueError("--obj is required for generation scenarios")

    files: dict[str, tuple[str, bytes, str]] = {
        "input_model": (
            args.obj.name,
            args.obj.read_bytes(),
            "text/plain",
        )
    }
    if args.style is not None:
        files["style_reference"] = (
            args.style.name,
            args.style.read_bytes(),
            "application/octet-stream",
        )
    return files


async def rest_health(args: argparse.Namespace) -> None:
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(f"{args.base_url}/health")
        print(response.status_code, response.json())
        response.raise_for_status()


async def rest_generate(args: argparse.Namespace) -> None:
    parameters = _generation_parameters(args)
    files = _rest_files(args)

    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream(
            "POST",
            f"{args.base_url}/generate",
            data=_rest_form_data(parameters),
            files=files,
        ) as response:
            print("HTTP status:", response.status_code)
            if response.status_code != 200:
                print((await response.aread()).decode(errors="replace"))
                return

            args.output.parent.mkdir(parents=True, exist_ok=True)
            total = 0
            with args.output.open("wb") as output_file:
                async for chunk in response.aiter_bytes():
                    output_file.write(chunk)
                    total += len(chunk)
            print(f"Saved {total} bytes to {args.output}")


async def rest_disconnect(args: argparse.Namespace) -> None:
    parameters = _generation_parameters(args)
    files = _rest_files(args)
    client = httpx.AsyncClient(timeout=None)
    request_task = asyncio.create_task(
        client.post(
            f"{args.base_url}/generate",
            data=_rest_form_data(parameters),
            files=files,
        )
    )

    await asyncio.sleep(args.cancel_after_seconds)
    print("Closing the HTTP client connection")
    request_task.cancel()
    with suppress(asyncio.CancelledError, Exception):
        await request_task
    await client.aclose()


async def rest_signal_cancel(args: argparse.Namespace) -> None:
    if args.server_pid is None:
        raise ValueError("--server-pid is required for signal cancellation")

    parameters = _generation_parameters(args)
    files = _rest_files(args)
    async with httpx.AsyncClient(timeout=None) as client:
        request_task = asyncio.create_task(
            client.post(
                f"{args.base_url}/generate",
                data=_rest_form_data(parameters),
                files=files,
            )
        )
        await asyncio.sleep(args.cancel_after_seconds)
        print(f"Sending SIGINT to server PID {args.server_pid}")
        os.kill(args.server_pid, signal.SIGINT)
        response = await request_task
        print("HTTP status:", response.status_code)
        print(response.text)


async def _send_file_chunks(websocket: Any, data: bytes, chunk_size: int) -> None:
    for offset in range(0, len(data), chunk_size):
        await websocket.send(data[offset : offset + chunk_size])


async def _open_generation_websocket(
    args: argparse.Namespace,
) -> tuple[Any, dict[str, Any]]:
    if args.obj is None:
        raise ValueError("--obj is required for generation scenarios")

    obj_bytes = args.obj.read_bytes()
    style_bytes = args.style.read_bytes() if args.style is not None else None
    parameters = _generation_parameters(args)

    start_message: dict[str, Any] = {
        "type": "generate",
        "parameters": parameters,
        "input_model": {
            "filename": args.obj.name,
            "size": len(obj_bytes),
        },
        "style_reference": (
            {
                "filename": args.style.name,
                "size": len(style_bytes),
            }
            if args.style is not None and style_bytes is not None
            else None
        ),
    }

    websocket = await websockets.connect(
        f"{_websocket_base_url(args.base_url)}/ws/generate",
        max_size=None,
        ping_interval=20,
        ping_timeout=20,
        close_timeout=5,
    )
    await websocket.send(json.dumps(start_message))

    upload_ready = json.loads(await websocket.recv())
    if upload_ready.get("type") != "upload_ready":
        raise RuntimeError(f"Unexpected server message: {upload_ready}")

    chunk_size = int(upload_ready.get("chunk_size", CLIENT_CHUNK_SIZE))
    await _send_file_chunks(websocket, obj_bytes, chunk_size)
    if style_bytes is not None:
        await _send_file_chunks(websocket, style_bytes, chunk_size)

    upload_received = json.loads(await websocket.recv())
    if upload_received.get("type") != "upload_received":
        raise RuntimeError(f"Unexpected server message: {upload_received}")

    return websocket, parameters


async def ws_health(args: argparse.Namespace) -> None:
    async with websockets.connect(
        f"{_websocket_base_url(args.base_url)}/ws/health",
        max_size=None,
    ) as websocket:
        print(json.loads(await websocket.recv()))


async def ws_generate(
    args: argparse.Namespace,
    *,
    cancel_after_progress: int | None = None,
    disconnect_after_seconds: float | None = None,
    signal_after_seconds: float | None = None,
) -> None:
    websocket, _ = await _open_generation_websocket(args)

    async def delayed_disconnect() -> None:
        assert disconnect_after_seconds is not None
        await asyncio.sleep(disconnect_after_seconds)
        print("Closing WebSocket connection")
        await websocket.close()

    async def delayed_signal() -> None:
        assert signal_after_seconds is not None
        if args.server_pid is None:
            raise ValueError("--server-pid is required for signal cancellation")
        await asyncio.sleep(signal_after_seconds)
        print(f"Sending SIGINT to server PID {args.server_pid}")
        os.kill(args.server_pid, signal.SIGINT)

    helper_tasks: list[asyncio.Task[None]] = []
    if disconnect_after_seconds is not None:
        helper_tasks.append(asyncio.create_task(delayed_disconnect()))
    if signal_after_seconds is not None:
        helper_tasks.append(asyncio.create_task(delayed_signal()))

    result_file = None
    received_size = 0
    expected_size: int | None = None
    cancel_sent = False

    try:
        while True:
            message = await websocket.recv()

            if isinstance(message, bytes):
                if result_file is None:
                    raise RuntimeError("Received binary data before result_ready")
                result_file.write(message)
                received_size += len(message)
                continue

            payload = json.loads(message)
            print(payload)
            message_type = payload.get("type")

            if message_type == "progress" and cancel_after_progress is not None:
                if int(payload.get("completed", 0)) >= cancel_after_progress and not cancel_sent:
                    print("Sending WebSocket cancellation message")
                    await websocket.send(json.dumps({"type": "cancel"}))
                    cancel_sent = True

            if message_type == "result_ready":
                expected_size = int(payload["size"])
                args.output.parent.mkdir(parents=True, exist_ok=True)
                result_file = args.output.open("wb")

            if message_type == "completed":
                if result_file is not None:
                    result_file.close()
                    result_file = None
                if expected_size is not None and received_size != expected_size:
                    raise RuntimeError(
                        f"Result size mismatch: expected {expected_size}, got {received_size}"
                    )
                print(f"Saved {received_size} bytes to {args.output}")
                return

            if message_type in {"cancelled", "error"}:
                return
    except websockets.ConnectionClosed as error:
        print(f"WebSocket closed: code={error.code}, reason={error.reason}")
    finally:
        if result_file is not None:
            result_file.close()
        for task in helper_tasks:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        with suppress(Exception):
            await websocket.close()


async def run(args: argparse.Namespace) -> None:
    scenarios = {
        "rest-health": rest_health,
        "rest-generate": rest_generate,
        "rest-disconnect": rest_disconnect,
        "rest-signal-cancel": rest_signal_cancel,
        "ws-health": ws_health,
    }

    if args.scenario in scenarios:
        await scenarios[args.scenario](args)
        return
    if args.scenario == "ws-generate":
        await ws_generate(args)
        return
    if args.scenario == "ws-cancel":
        await ws_generate(args, cancel_after_progress=args.cancel_after_progress)
        return
    if args.scenario == "ws-disconnect":
        await ws_generate(args, disconnect_after_seconds=args.cancel_after_seconds)
        return
    if args.scenario == "ws-signal-cancel":
        await ws_generate(args, signal_after_seconds=args.cancel_after_seconds)
        return
    raise ValueError(f"Unknown scenario: {args.scenario}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exercise the REST and WebSocket generation APIs.")
    parser.add_argument(
        "scenario",
        choices=[
            "rest-health",
            "rest-generate",
            "rest-disconnect",
            "rest-signal-cancel",
            "ws-health",
            "ws-generate",
            "ws-cancel",
            "ws-disconnect",
            "ws-signal-cancel",
        ],
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--obj", type=Path)
    parser.add_argument("--style", type=Path)
    parser.add_argument("--output", type=Path, default=Path("result.glb"))
    parser.add_argument("--remote-output-filename", default="building.glb")
    parser.add_argument("--pixels-per-meter", type=int, default=32)
    parser.add_argument("--cluster-count", type=int, default=4)
    parser.add_argument(
        "--parameters-json",
        help="JSON object or path to a JSON file with generation parameter overrides.",
    )
    parser.add_argument("--cancel-after-progress", type=int, default=2)
    parser.add_argument("--cancel-after-seconds", type=float, default=30.0)
    parser.add_argument("--server-pid", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
