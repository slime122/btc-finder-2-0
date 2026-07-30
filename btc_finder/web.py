"""FastAPI dashboard for controlling an isolated BTC Puzzle worker process."""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import queue
import sys
import threading
import time
import webbrowser
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT / "web" / "static"
GPU_WATCHDOG_BATCH_LIMIT = 512
GPU_SAFE_BATCH_SIZE = 256
CPU_DEFAULT_BATCH_SIZE = 4096


class StartRequest(BaseModel):
    user_token: str = Field(..., min_length=1)
    worker_name: str = Field("macbook", min_length=1, max_length=15)
    puzzle_code: int = Field(71, ge=1)
    hardware: Literal["gpu", "cpu"] = "cpu"
    batch_size: int = Field(CPU_DEFAULT_BATCH_SIZE, ge=0)
    base_url: str = "https://api.btcpuzzle.info"


@dataclass
class DashboardState:
    status: str = "Idle"
    running: bool = False
    hardware: str = "cpu"
    device_name: str = "-"
    puzzle_code: int = 71
    batch_size: int = CPU_DEFAULT_BATCH_SIZE
    current_hex: str = "-"
    scanned: int = 0
    progress: float = 0.0
    range_total: int = 0
    speed: float = 0.0
    speed_keys_s: float = 0.0
    proofs_found: int = 0
    pow_found: int = 0
    last_error: Optional[str] = None
    started_at: Optional[float] = None
    worker_exit_code: Optional[int] = None


class WorkerController:
    def __init__(self) -> None:
        self._ctx = mp.get_context("spawn")
        self._lock = threading.Lock()
        self._state = DashboardState()
        self._logs: deque[str] = deque(maxlen=350)
        self._process: Optional[mp.Process] = None
        self._event_queue: Optional[mp.Queue] = None
        self._stop_event: Optional[mp.Event] = None
        self._monitor_thread: Optional[threading.Thread] = None

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            payload = asdict(self._state)
            payload["speed_label"] = _format_speed(self._state.speed)
            payload["logs"] = list(self._logs)
            return payload

    def start(self, request: StartRequest) -> dict[str, Any]:
        normalized = self._normalize_request(request)
        with self._lock:
            self._reap_finished_process_locked()
            if self._state.running:
                raise HTTPException(status_code=409, detail="Worker already running")

            self._event_queue = self._ctx.Queue()
            self._stop_event = self._ctx.Event()
            self._process = self._ctx.Process(
                target=_worker_process_main,
                args=(_model_to_dict(normalized), self._event_queue, self._stop_event),
                name="btc-finder-opencl-worker",
                daemon=True,
            )
            self._state = DashboardState(
                status="Starting",
                running=True,
                hardware=normalized.hardware,
                puzzle_code=normalized.puzzle_code,
                batch_size=normalized.batch_size,
                started_at=time.time(),
            )
            self._logs.clear()

        self._log("Starting isolated worker process")
        if normalized.batch_size != request.batch_size:
            self._log(
                f"Batch adjusted from {request.batch_size} to {normalized.batch_size} "
                f"for {normalized.hardware.upper()} safety"
            )
        self._process.start()
        self._monitor_thread = threading.Thread(
            target=self._monitor_worker,
            name="btc-finder-process-monitor",
            daemon=True,
        )
        self._monitor_thread.start()
        return self.snapshot()

    def stop(self) -> dict[str, Any]:
        with self._lock:
            process = self._process
            stop_event = self._stop_event
            running = self._state.running
            if running:
                self._state.status = "Stopping"
        if stop_event:
            stop_event.set()
        if running:
            self._log("Stop requested")
        if process and process.is_alive():
            threading.Thread(
                target=self._terminate_if_needed,
                args=(process,),
                name="btc-finder-stop-watch",
                daemon=True,
            ).start()
        return self.snapshot()

    def devices(self) -> list[dict[str, Any]]:
        from .opencl_kernel import OpenCLScanner

        return [asdict(device) for device in OpenCLScanner.list_devices()]

    def _normalize_request(self, request: StartRequest) -> StartRequest:
        batch_size = request.batch_size
        if request.hardware == "gpu":
            if batch_size <= 0:
                batch_size = GPU_SAFE_BATCH_SIZE
            if sys.platform == "darwin" and batch_size > GPU_WATCHDOG_BATCH_LIMIT:
                batch_size = GPU_SAFE_BATCH_SIZE
        elif batch_size <= 0:
            batch_size = CPU_DEFAULT_BATCH_SIZE
        return _model_copy(request, {"batch_size": batch_size})

    def _monitor_worker(self) -> None:
        process = self._process
        event_queue = self._event_queue
        if not process or not event_queue:
            return

        while process.is_alive() or not event_queue.empty():
            try:
                event = event_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            self._apply_worker_event(event)

        process.join(timeout=0.1)
        exit_code = process.exitcode
        with self._lock:
            self._state.worker_exit_code = exit_code
            self._state.running = False
            if self._state.status not in ("Found Key",):
                self._state.status = "Idle" if exit_code in (0, None) else "Worker Crashed"
            if exit_code not in (0, None) and not self._state.last_error:
                self._state.last_error = f"Worker process exited with code {exit_code}"
        if exit_code not in (0, None):
            self._log(
                "Worker process crashed or was aborted. Dashboard stayed online. "
                f"Exit code: {exit_code}"
            )
        else:
            self._log("Worker stopped")

    def _apply_worker_event(self, event: dict[str, Any]) -> None:
        event_type = event.get("type")
        if event_type == "log":
            self._log(str(event.get("message", "")))
            return

        with self._lock:
            if event_type == "status":
                self._state.status = str(event.get("status", self._state.status))
            elif event_type == "device":
                self._state.device_name = str(event.get("name", "-"))
                self._state.batch_size = int(event.get("batch_size", self._state.batch_size))
            elif event_type == "range":
                self._state.current_hex = str(event.get("hex", "-"))
                self._state.range_total = int(event.get("total", 0))
                self._state.scanned = 0
                self._state.progress = 0.0
                self._state.speed = 0.0
                self._state.speed_keys_s = 0.0
                self._state.proofs_found = 0
                self._state.pow_found = 0
            elif event_type == "metrics":
                speed = float(event.get("speed", 0.0))
                progress = float(event.get("progress", 0.0))
                proofs = int(event.get("proofs_found", 0))
                self._state.scanned = int(event.get("scanned", self._state.scanned))
                self._state.speed = speed
                self._state.speed_keys_s = speed
                self._state.progress = max(0.0, min(1.0, progress))
                self._state.proofs_found = proofs
                self._state.pow_found = proofs
            elif event_type == "error":
                self._state.last_error = str(event.get("message", "Unknown worker error"))
                self._state.status = "Error"
            elif event_type == "found":
                self._state.status = "Found Key"
                self._state.running = False

    def _set_status(self, status: str) -> None:
        with self._lock:
            self._state.status = status

    def _log(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        with self._lock:
            self._logs.append(f"[{timestamp}] {message}")

    def _reap_finished_process_locked(self) -> None:
        if self._process and not self._process.is_alive():
            self._process.join(timeout=0.1)
            self._state.running = False

    def _terminate_if_needed(self, process: mp.Process) -> None:
        process.join(timeout=8.0)
        if process.is_alive():
            self._log("Worker did not stop cleanly; terminating process")
            process.terminate()
            process.join(timeout=2.0)
        if process.is_alive() and hasattr(process, "kill"):
            self._log("Worker ignored terminate; killing process")
            process.kill()
            process.join(timeout=1.0)


def _worker_process_main(
    request_data: dict[str, Any],
    event_queue: mp.Queue,
    stop_event: mp.Event,
) -> None:
    try:
        from .api import BTCPuzzleAPI
        from .opencl_kernel import OpenCLScanner, ScanRequest, ScanResult

        request = StartRequest(**request_data)
        _emit(event_queue, "status", status="Initializing OpenCL")
        scanner = OpenCLScanner(
            prefer_gpu=request.hardware == "gpu",
            force_cpu=request.hardware == "cpu",
        )
        device = scanner.device_info
        batch_size = request.batch_size or scanner.safe_default_batch_size
        if request.hardware == "gpu" and sys.platform == "darwin" and batch_size > GPU_WATCHDOG_BATCH_LIMIT:
            _emit(
                event_queue,
                "log",
                message=(
                    f"GPU batch {batch_size} exceeds macOS watchdog guard; "
                    f"forcing {GPU_SAFE_BATCH_SIZE}"
                ),
            )
            batch_size = GPU_SAFE_BATCH_SIZE
        _emit(event_queue, "device", name=device.name, batch_size=batch_size)
        _emit(
            event_queue,
            "log",
            message=f"OpenCL device: {device.name} ({device.type}, {device.compute_units} CUs)",
        )
        _emit(event_queue, "log", message=f"Batch size: {batch_size}")

        api = BTCPuzzleAPI(
            user_token=request.user_token,
            worker_name=request.worker_name,
            base_url=request.base_url,
        )

        while not stop_event.is_set():
            _emit(event_queue, "status", status="Requesting range")
            _emit(event_queue, "log", message=f"GET /puzzle/{request.puzzle_code}/range")
            workload = api.get_range(request.puzzle_code)
            _emit(
                event_queue,
                "range",
                hex=workload.hex,
                total=workload.key_count,
            )
            _emit(
                event_queue,
                "log",
                message=(
                    f"Range {workload.hex}: {workload.key_count:,} keys "
                    f"({workload.workload_start}..{workload.workload_end})"
                ),
            )
            _emit(event_queue, "status", status="Running")

            ping_stop = threading.Event()
            ping_thread = threading.Thread(
                target=_ping_loop,
                args=(api, request.puzzle_code, workload.hex, event_queue, stop_event, ping_stop),
                daemon=True,
            )
            ping_thread.start()

            last_scanned = 0
            last_sample_time = time.monotonic()

            def progress_callback(result: ScanResult) -> None:
                nonlocal last_scanned, last_sample_time
                now = time.monotonic()
                elapsed = max(now - last_sample_time, 1e-9)
                scanned_delta = max(result.scanned - last_scanned, 0)
                speed = scanned_delta / elapsed
                last_scanned = result.scanned
                last_sample_time = now
                proofs_found = len(result.proof_private_keys_hex or {})
                progress = result.scanned / workload.key_count if workload.key_count else 0.0
                _emit(
                    event_queue,
                    "metrics",
                    scanned=result.scanned,
                    speed=speed,
                    progress=progress,
                    proofs_found=proofs_found,
                )

            try:
                result = scanner.scan(
                    ScanRequest(
                        start_private_key_hex=workload.start_private_key_hex,
                        end_private_key_hex=workload.end_private_key_hex,
                        target_address=workload.target_address,
                        proof_addresses=workload.proof_of_work_addresses,
                        batch_size=batch_size,
                    ),
                    progress_callback=progress_callback,
                    stop_event=stop_event,
                )
            finally:
                ping_stop.set()
                ping_thread.join(timeout=0.2)

            if result.target_private_key_hex:
                path = PROJECT_ROOT / "FOUND_KEY.txt"
                path.write_text(
                    "target_address="
                    f"{workload.target_address}\nprivate_key_hex="
                    f"{result.target_private_key_hex}\n",
                    encoding="utf-8",
                )
                _emit(event_queue, "found")
                _emit(event_queue, "log", message=f"FOUND_KEY saved to {path}")
                return

            if stop_event.is_set():
                break

            proof_keys = result.proof_private_keys_hex or {}
            if len(proof_keys) != 6:
                _emit(event_queue, "log", message="Range skipped: PoW keys incomplete, PUT not sent")
                continue

            _emit(event_queue, "status", status="Submitting PUT")
            _emit(event_queue, "log", message=f"PUT /puzzle/{request.puzzle_code}/range HEX={workload.hex}")
            ordered = [proof_keys[address] for address in workload.proof_of_work_addresses]
            response = api.submit_range(
                puzzle_code=request.puzzle_code,
                workload_hex=workload.hex,
                proof_keys=ordered,
                gpu_count=0 if request.hardware == "cpu" else 1,
            )
            _emit(event_queue, "log", message=f"PUT response: {response}")

        _emit(event_queue, "status", status="Idle")
    except BaseException as exc:
        _emit(event_queue, "error", message=f"{type(exc).__name__}: {exc}")
        _emit(event_queue, "log", message=f"ERROR: {type(exc).__name__}: {exc}")
        raise


def _ping_loop(
    api: Any,
    puzzle_code: int,
    workload_hex: str,
    event_queue: mp.Queue,
    stop_event: mp.Event,
    ping_stop: threading.Event,
    interval_seconds: int = 150,
) -> None:
    while not stop_event.is_set():
        if ping_stop.wait(interval_seconds):
            return
        try:
            _emit(event_queue, "status", status="Pinging")
            _emit(event_queue, "log", message=f"PATCH /puzzle/{puzzle_code}/range/ping HEX={workload_hex}")
            api.ping(puzzle_code, workload_hex)
            _emit(event_queue, "log", message="Ping OK")
            _emit(event_queue, "status", status="Running")
        except Exception as exc:
            _emit(event_queue, "log", message=f"Ping failed: {type(exc).__name__}: {exc}")
            _emit(event_queue, "status", status="Running")


def _emit(event_queue: mp.Queue, event_type: str, **payload: Any) -> None:
    payload["type"] = event_type
    event_queue.put(payload)


def _format_speed(keys_per_second: float) -> str:
    if keys_per_second >= 1_000_000:
        return f"{keys_per_second / 1_000_000:.2f} Mkeys/s"
    if keys_per_second >= 1_000:
        return f"{keys_per_second / 1_000:.0f} Kkeys/s"
    return f"{keys_per_second:.0f} keys/s"


def _model_to_dict(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _model_copy(model: StartRequest, update: dict[str, Any]) -> StartRequest:
    if hasattr(model, "model_copy"):
        return model.model_copy(update=update)
    return model.copy(update=update)


controller = WorkerController()
app = FastAPI(title="BTC Finder 2.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("shutdown")
def shutdown_worker() -> None:
    controller.stop()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/status")
def status() -> dict[str, Any]:
    return controller.snapshot()


@app.get("/api/devices")
def devices() -> dict[str, Any]:
    try:
        return {"devices": controller.devices()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/start")
def start(request: StartRequest) -> dict[str, Any]:
    return controller.start(request)


@app.post("/api/stop")
def stop() -> dict[str, Any]:
    return controller.stop()


@app.websocket("/ws")
async def websocket_status(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(controller.snapshot())
            await asyncio.sleep(0.5)
    except (WebSocketDisconnect, asyncio.CancelledError):
        return


def run_web_server(host: str = "127.0.0.1", port: int = 3000, open_browser: bool = True) -> None:
    import uvicorn

    url = f"http://{host}:{port}"
    print(f"BTC Finder dashboard: {url}")
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=host, port=port, log_level="info")
