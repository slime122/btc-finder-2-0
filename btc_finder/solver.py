"""Continuous BTC Puzzle worker orchestration."""

from __future__ import annotations

import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .api import BTCPuzzleAPI
from .opencl_kernel import DEFAULT_BATCH_SIZE, OpenCLScanner, ScanRequest, ScanResult


FOUND_KEY_PATH = Path("FOUND_KEY.txt")


@dataclass(frozen=True)
class SolverConfig:
    puzzle_code: int = 71
    batch_size: int = DEFAULT_BATCH_SIZE
    ping_interval_seconds: int = 150
    sleep_between_ranges_seconds: float = 2.0
    gpu_count: int = 1
    once: bool = False


class PuzzleSolver:
    """Coordinates API ranges, OpenCL scans, pings and final submissions."""

    def __init__(
        self,
        api: BTCPuzzleAPI,
        scanner: OpenCLScanner,
        config: SolverConfig,
    ) -> None:
        self.api = api
        self.scanner = scanner
        self.config = config
        self.stop_event = threading.Event()

    def install_signal_handlers(self) -> None:
        def handle_stop(signum, _frame) -> None:
            print(f"\nRecebido sinal {signum}. Encerrando ao fim do lote atual...")
            self.stop_event.set()

        signal.signal(signal.SIGINT, handle_stop)
        signal.signal(signal.SIGTERM, handle_stop)

    def run_forever(self) -> None:
        device = self.scanner.device_info
        print(
            "OpenCL ativo: "
            f"{device.name} ({device.type}, {device.compute_units} CUs, "
            f"{device.global_mem_mb} MB)"
        )

        while not self.stop_event.is_set():
            workload = self.api.get_range(self.config.puzzle_code)
            print(
                f"\nRange recebido: HEX={workload.hex} "
                f"keys={workload.key_count:,} "
                f"start={workload.start_private_key_hex} "
                f"end={workload.end_private_key_hex}"
            )

            ping_thread, ping_stop_event = self._start_ping_thread(workload.hex)
            last_print = 0.0

            def progress(result: ScanResult) -> None:
                nonlocal last_print
                now = time.monotonic()
                if now - last_print < 1.0:
                    return
                last_print = now
                proof_found = len(result.proof_private_keys_hex or {})
                speed = result.current_speed_keys_s or result.keys_per_second
                print(
                    "\r"
                    f"Escaneadas: {result.scanned:,} | "
                    f"{_format_speed(speed)} | "
                    f"PoW: {proof_found}/6",
                    end="",
                    flush=True,
                )

            try:
                result = self.scanner.scan(
                    ScanRequest(
                        start_private_key_hex=workload.start_private_key_hex,
                        end_private_key_hex=workload.end_private_key_hex,
                        target_address=workload.target_address,
                        proof_addresses=workload.proof_of_work_addresses,
                        batch_size=self.config.batch_size,
                    ),
                    progress_callback=progress,
                    stop_event=self.stop_event,
                )
            finally:
                ping_stop_event.set()
                if self.config.once:
                    self.stop_event.set()
                ping_thread.join(timeout=0.1)

            print()
            proof_keys = result.proof_private_keys_hex or {}
            if result.target_private_key_hex:
                self._write_found_key(result.target_private_key_hex, workload.target_address)
                return

            if len(proof_keys) != 6:
                print(
                    "Range não confirmado: as 6 chaves de prova não foram encontradas. "
                    "Nada foi enviado ao PUT."
                )
                if self.config.once:
                    return
                time.sleep(self.config.sleep_between_ranges_seconds)
                continue

            ordered_proof_keys = [
                proof_keys[address] for address in workload.proof_of_work_addresses
            ]
            response = self.api.submit_range(
                puzzle_code=self.config.puzzle_code,
                workload_hex=workload.hex,
                proof_keys=ordered_proof_keys,
                gpu_count=self.config.gpu_count,
            )
            print(f"Range {workload.hex} confirmado na pool: {response}")

            if self.config.once:
                return
            time.sleep(self.config.sleep_between_ranges_seconds)

    def _start_ping_thread(self, workload_hex: str) -> tuple[threading.Thread, threading.Event]:
        ping_stop_event = threading.Event()

        def ping_loop() -> None:
            while not self.stop_event.is_set():
                if ping_stop_event.wait(self.config.ping_interval_seconds):
                    return
                try:
                    self.api.ping(self.config.puzzle_code, workload_hex)
                    print(f"\nPing enviado para HEX={workload_hex}")
                except Exception as exc:
                    print(f"\nFalha no ping para HEX={workload_hex}: {exc}")

        thread = threading.Thread(target=ping_loop, name="btcpuzzle-ping", daemon=True)
        thread.start()
        return thread, ping_stop_event

    @staticmethod
    def _write_found_key(private_key_hex: str, target_address: str) -> None:
        FOUND_KEY_PATH.write_text(
            f"target_address={target_address}\nprivate_key_hex={private_key_hex}\n",
            encoding="utf-8",
        )
        print("\n" + "=" * 72)
        print("CHAVE PRINCIPAL ENCONTRADA")
        print(f"Endereço: {target_address}")
        print(f"Private key HEX: {private_key_hex}")
        print(f"Arquivo salvo: {FOUND_KEY_PATH.resolve()}")
        print("=" * 72)


def _format_speed(keys_per_second: float) -> str:
    if keys_per_second >= 1_000_000:
        return f"{keys_per_second / 1_000_000:.3f} Mkeys/s"
    if keys_per_second >= 1_000:
        return f"{keys_per_second / 1_000:.3f} Kkeys/s"
    return f"{keys_per_second:.0f} keys/s"
