"""PyOpenCL device selection and batched scan execution."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np


BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
DEFAULT_KERNEL_PATH = Path(__file__).resolve().parent.parent / "kernels" / "secp256k1.cl"


class OpenCLScannerError(RuntimeError):
    """Raised when OpenCL cannot compile or execute the scan kernel."""


@dataclass(frozen=True)
class DeviceInfo:
    platform: str
    name: str
    type: str
    compute_units: int
    global_mem_mb: int
    max_work_group_size: int


@dataclass(frozen=True)
class ScanRequest:
    start_private_key_hex: str
    end_private_key_hex: str
    target_address: str
    proof_addresses: tuple[str, ...]
    batch_size: int


@dataclass
class ScanResult:
    scanned: int = 0
    elapsed_seconds: float = 0.0
    target_private_key_hex: Optional[str] = None
    proof_private_keys_hex: dict[str, str] | None = None

    @property
    def keys_per_second(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.scanned / self.elapsed_seconds

    @property
    def mkeys_per_second(self) -> float:
        return self.keys_per_second / 1_000_000


class OpenCLScanner:
    """Owns the OpenCL context, command queue, compiled kernel and scan loop."""

    def __init__(
        self,
        kernel_path: Path | str = DEFAULT_KERNEL_PATH,
        prefer_gpu: bool = True,
        platform_hint: Optional[str] = None,
        device_hint: Optional[str] = None,
        local_size: int = 128,
    ) -> None:
        self.kernel_path = Path(kernel_path)
        self.platform, self.device = self._select_device(prefer_gpu, platform_hint, device_hint)
        self.local_size = min(local_size, int(self.device.max_work_group_size))
        self.context = cl.Context([self.device])
        self.queue = cl.CommandQueue(
            self.context,
            properties=cl.command_queue_properties.PROFILING_ENABLE,
        )
        self.program = self._build_program()
        self.kernel = cl.Kernel(self.program, "scan_range")

    @property
    def device_info(self) -> DeviceInfo:
        return DeviceInfo(
            platform=self.platform.name,
            name=self.device.name,
            type=cl.device_type.to_string(self.device.type),
            compute_units=self.device.max_compute_units,
            global_mem_mb=self.device.global_mem_size // (1024 * 1024),
            max_work_group_size=self.device.max_work_group_size,
        )

    @staticmethod
    def list_devices() -> list[DeviceInfo]:
        devices: list[DeviceInfo] = []
        for platform in cl.get_platforms():
            for device in platform.get_devices():
                devices.append(
                    DeviceInfo(
                        platform=platform.name,
                        name=device.name,
                        type=cl.device_type.to_string(device.type),
                        compute_units=device.max_compute_units,
                        global_mem_mb=device.global_mem_size // (1024 * 1024),
                        max_work_group_size=device.max_work_group_size,
                    )
                )
        return devices

    def scan(
        self,
        request: ScanRequest,
        progress_callback=None,
        stop_event=None,
    ) -> ScanResult:
        """Scan a private-key range in bounded OpenCL batches."""

        if len(request.proof_addresses) != 6:
            raise ValueError("BTC Puzzle API always expects exactly 6 proof addresses")

        start_int = int(_normalize_private_key_hex(request.start_private_key_hex), 16)
        end_int = int(_normalize_private_key_hex(request.end_private_key_hex), 16)
        if end_int < start_int:
            raise ValueError("end_private_key_hex must be >= start_private_key_hex")

        target_hash160 = address_to_hash160(request.target_address)
        proof_hash160s = b"".join(address_to_hash160(address) for address in request.proof_addresses)
        max_keys = end_int - start_int + 1
        batch_size = max(1, min(request.batch_size, max_keys))

        target_hash_buf = cl.Buffer(
            self.context,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=np.frombuffer(target_hash160, dtype=np.uint8),
        )
        proof_hash_buf = cl.Buffer(
            self.context,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=np.frombuffer(proof_hash160s, dtype=np.uint8),
        )
        found_flags = cl_array.zeros(self.queue, 7, dtype=np.uint32)
        found_keys = cl_array.zeros(self.queue, 7 * 8, dtype=np.uint32)

        proof_map: dict[str, str] = {}
        target_private_key: Optional[str] = None
        scanned = 0
        started = time.perf_counter()

        while scanned < max_keys:
            if stop_event and stop_event.is_set():
                break

            current = start_int + scanned
            current_batch = min(batch_size, max_keys - scanned)
            global_size = _round_up(current_batch, self.local_size)
            limbs = _int_to_uint32_limbs(current)

            event = self.kernel(
                self.queue,
                (global_size,),
                (self.local_size,),
                np.uint32(limbs[0]),
                np.uint32(limbs[1]),
                np.uint32(limbs[2]),
                np.uint32(limbs[3]),
                np.uint32(limbs[4]),
                np.uint32(limbs[5]),
                np.uint32(limbs[6]),
                np.uint32(limbs[7]),
                np.uint64(current_batch),
                target_hash_buf,
                proof_hash_buf,
                found_flags.data,
                found_keys.data,
            )
            event.wait()
            scanned += current_batch

            flags = found_flags.get()
            key_words = found_keys.get()
            for index, is_found in enumerate(flags):
                if not is_found:
                    continue
                key_hex = _uint32_limbs_to_hex(key_words[index * 8 : (index + 1) * 8])
                if index == 0:
                    target_private_key = key_hex
                else:
                    proof_map[request.proof_addresses[index - 1]] = key_hex

            elapsed = time.perf_counter() - started
            if progress_callback:
                progress_callback(
                    ScanResult(
                        scanned=scanned,
                        elapsed_seconds=elapsed,
                        target_private_key_hex=target_private_key,
                        proof_private_keys_hex=dict(proof_map),
                    )
                )
            if target_private_key or len(proof_map) == 6:
                if len(proof_map) == 6 or target_private_key:
                    break

        return ScanResult(
            scanned=scanned,
            elapsed_seconds=time.perf_counter() - started,
            target_private_key_hex=target_private_key,
            proof_private_keys_hex=proof_map,
        )

    def _build_program(self) -> cl.Program:
        if not self.kernel_path.exists():
            raise OpenCLScannerError(f"Kernel file not found: {self.kernel_path}")
        source = self.kernel_path.read_text(encoding="utf-8")
        try:
            return cl.Program(self.context, source).build(
                options=["-cl-std=CL1.2", "-Werror"]
            )
        except Exception as exc:
            raise OpenCLScannerError(f"Failed to build OpenCL kernel: {exc}") from exc

    @staticmethod
    def _select_device(
        prefer_gpu: bool,
        platform_hint: Optional[str],
        device_hint: Optional[str],
    ) -> tuple[cl.Platform, cl.Device]:
        matches: list[tuple[int, cl.Platform, cl.Device]] = []
        for platform in cl.get_platforms():
            if platform_hint and platform_hint.lower() not in platform.name.lower():
                continue
            for device in platform.get_devices():
                if device_hint and device_hint.lower() not in device.name.lower():
                    continue
                is_gpu = bool(device.type & cl.device_type.GPU)
                is_cpu = bool(device.type & cl.device_type.CPU)
                score = device.max_compute_units
                if prefer_gpu and is_gpu:
                    score += 10_000
                elif is_cpu:
                    score += 1_000
                matches.append((score, platform, device))

        if not matches:
            raise OpenCLScannerError("No OpenCL devices found")
        _, platform, device = max(matches, key=lambda item: item[0])
        return platform, device


def address_to_hash160(address: str) -> bytes:
    """Decode a legacy Base58Check Bitcoin address and return its hash160."""

    raw = _base58check_decode(address)
    if len(raw) != 21:
        raise ValueError(f"Unsupported Bitcoin address payload length: {address}")
    version = raw[0]
    if version not in (0x00,):
        raise ValueError(f"Only legacy P2PKH mainnet addresses are supported: {address}")
    return raw[1:]


def _base58check_decode(value: str) -> bytes:
    number = 0
    for char in value:
        number *= 58
        if char not in BASE58_ALPHABET:
            raise ValueError(f"Invalid Base58 character {char!r} in {value!r}")
        number += BASE58_ALPHABET.index(char)

    leading_zeroes = len(value) - len(value.lstrip("1"))
    payload = b"\x00" * leading_zeroes + number.to_bytes((number.bit_length() + 7) // 8, "big")
    if len(payload) < 5:
        raise ValueError(f"Invalid Base58Check payload: {value!r}")
    body, checksum = payload[:-4], payload[-4:]
    import hashlib

    expected = hashlib.sha256(hashlib.sha256(body).digest()).digest()[:4]
    if checksum != expected:
        raise ValueError(f"Invalid Base58Check checksum for address {value!r}")
    return body


def _normalize_private_key_hex(value: str) -> str:
    cleaned = value.strip().lower().removeprefix("0x")
    int(cleaned, 16)
    if len(cleaned) > 64:
        raise ValueError("private key hex cannot be longer than 64 chars")
    return cleaned.zfill(64)


def _int_to_uint32_limbs(value: int) -> list[int]:
    if value < 0 or value >= 1 << 256:
        raise ValueError("private key integer must fit in 256 bits")
    return [(value >> shift) & 0xFFFFFFFF for shift in range(224, -1, -32)]


def _uint32_limbs_to_hex(words) -> str:
    return "".join(f"{int(word):08x}" for word in words)


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple
