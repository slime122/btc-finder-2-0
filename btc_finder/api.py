"""REST client for the btcpuzzle.info solo pool API."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Optional

import requests


DEFAULT_BASE_URL = "https://api.btcpuzzle.info"


class PuzzleAPIError(RuntimeError):
    """Raised when the pool API returns an error or malformed payload."""


@dataclass(frozen=True)
class PuzzleRange:
    """A workload range returned by the pool."""

    hex: str
    puzzle_code: int
    workload_start: str
    workload_end: str
    target_address: str
    proof_of_work_addresses: tuple[str, ...]

    @property
    def start_private_key_hex(self) -> str:
        return _pad_private_key_hex(f"{self.hex}{self.workload_start}")

    @property
    def end_private_key_hex(self) -> str:
        return _pad_private_key_hex(f"{self.hex}{self.workload_end}")

    @property
    def key_count(self) -> int:
        start = int(self.start_private_key_hex, 16)
        end = int(self.end_private_key_hex, 16)
        return max(0, end - start + 1)


class BTCPuzzleAPI:
    """Small typed wrapper around the btcpuzzle.info pool endpoints."""

    def __init__(
        self,
        user_token: str,
        worker_name: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 30.0,
        session: Optional[requests.Session] = None,
    ) -> None:
        if not user_token:
            raise ValueError("user_token is required")
        if worker_name and len(worker_name) > 15:
            raise ValueError("worker_name must be at most 15 characters")

        self.user_token = user_token
        self.worker_name = worker_name
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()

    def get_range(self, puzzle_code: int = 71, custom_range: Optional[str] = None) -> PuzzleRange:
        """Request a new range from the pool."""

        headers = self._base_headers(include_worker=False)
        if self.worker_name:
            headers["WorkerName"] = self.worker_name
        if custom_range:
            headers["CustomRange"] = custom_range

        response = self.session.get(
            self._url(puzzle_code, "range"),
            headers=headers,
            timeout=self.timeout,
        )
        payload = self._json_response(response)

        try:
            proof_addresses = tuple(payload["proofOfWorkAddresses"])
            return PuzzleRange(
                hex=str(payload["hex"]).upper(),
                puzzle_code=int(payload.get("puzzleCode", puzzle_code)),
                workload_start=str(payload["workloadStart"]).upper(),
                workload_end=str(payload["workloadEnd"]).upper(),
                target_address=str(payload["targetAddress"]),
                proof_of_work_addresses=proof_addresses,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise PuzzleAPIError(f"Malformed range payload: {payload!r}") from exc

    def submit_range(
        self,
        puzzle_code: int,
        workload_hex: str,
        proof_keys: Iterable[str],
        gpu_count: int = 1,
    ) -> dict:
        """Mark a range as scanned after all proof private keys are found."""

        hashed_proof_key = hash_proof_keys(proof_keys)
        headers = self._base_headers(include_worker=True)
        headers.update(
            {
                "HEX": workload_hex.upper(),
                "HashedProofKey": hashed_proof_key,
                "GPUCount": str(gpu_count),
            }
        )

        response = self.session.put(
            self._url(puzzle_code, "range"),
            headers=headers,
            timeout=self.timeout,
        )
        return self._json_response(response)

    def ping(self, puzzle_code: int, workload_hex: str) -> bool:
        """Notify the pool that the current worker is still scanning a range."""

        headers = self._base_headers(include_worker=True)
        headers["Hex"] = workload_hex.upper()
        response = self.session.patch(
            self._url(puzzle_code, "range/ping"),
            headers=headers,
            timeout=self.timeout,
        )
        if response.status_code == 200:
            return True
        raise PuzzleAPIError(
            f"Ping failed with HTTP {response.status_code}: {response.text[:300]}"
        )

    def test_connection(self, puzzle_code: int = 38) -> bool:
        """Perform a cheap API reachability check by requesting a test-pool range."""

        self.get_range(puzzle_code=puzzle_code)
        return True

    def _base_headers(self, include_worker: bool) -> dict[str, str]:
        headers = {
            "UserToken": self.user_token,
            "Accept": "application/json",
            "User-Agent": "btc-finder-python-opencl/0.1.0",
        }
        if include_worker:
            if not self.worker_name:
                raise ValueError("worker_name is required for this API operation")
            headers["WorkerName"] = self.worker_name
        return headers

    def _url(self, puzzle_code: int, suffix: str) -> str:
        return f"{self.base_url}/puzzle/{puzzle_code}/{suffix.lstrip('/')}"

    @staticmethod
    def _json_response(response: requests.Response) -> dict:
        if not response.ok:
            raise PuzzleAPIError(
                f"API returned HTTP {response.status_code}: {response.text[:500]}"
            )
        try:
            payload = response.json()
        except ValueError as exc:
            raise PuzzleAPIError(f"API returned non-JSON response: {response.text[:500]}") from exc
        if isinstance(payload, dict) and payload.get("isSuccess") is False:
            raise PuzzleAPIError(f"API rejected request: {payload!r}")
        if not isinstance(payload, dict):
            raise PuzzleAPIError(f"API returned unexpected payload: {payload!r}")
        return payload


def hash_proof_keys(proof_keys: Iterable[str]) -> str:
    """Return SHA256(proofKey1+...+proofKey6) as lowercase hex."""

    normalized = [_pad_private_key_hex(key) for key in proof_keys]
    if len(normalized) != 6:
        raise ValueError("exactly 6 proof private keys are required")
    joined = "".join(normalized)
    return hashlib.sha256(joined.encode("ascii")).hexdigest()


def _pad_private_key_hex(value: str) -> str:
    cleaned = value.strip().lower().removeprefix("0x")
    if not cleaned:
        raise ValueError("private key hex cannot be empty")
    int(cleaned, 16)
    if len(cleaned) > 64:
        raise ValueError(f"private key hex is longer than 64 chars: {value!r}")
    return cleaned.zfill(64)
