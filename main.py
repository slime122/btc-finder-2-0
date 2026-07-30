"""CLI entrypoint for the modern BTC Puzzle OpenCL worker."""

from __future__ import annotations

import argparse
import os

from btc_finder.api import BTCPuzzleAPI
from btc_finder.opencl_kernel import OpenCLScanner
from btc_finder.solver import PuzzleSolver, SolverConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cliente Python/PyOpenCL para a Solo Pool BTC Puzzle."
    )
    parser.add_argument("--token", default=os.getenv("BTCPUZZLE_USER_TOKEN"))
    parser.add_argument("--worker", default=os.getenv("BTCPUZZLE_WORKER_NAME", "macbook"))
    parser.add_argument("--puzzle", type=int, default=int(os.getenv("BTCPUZZLE_PUZZLE", "71")))
    parser.add_argument("--base-url", default=os.getenv("BTCPUZZLE_API_URL", "https://api.btcpuzzle.info"))
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("BTCPUZZLE_BATCH_SIZE", "1048576")))
    parser.add_argument("--ping-interval", type=int, default=int(os.getenv("BTCPUZZLE_PING_INTERVAL", "150")))
    parser.add_argument("--gpu-count", type=int, default=int(os.getenv("BTCPUZZLE_GPU_COUNT", "1")))
    parser.add_argument("--platform")
    parser.add_argument("--device")
    parser.add_argument("--cpu", action="store_true", help="Permite selecionar CPU antes de GPU.")
    parser.add_argument("--once", action="store_true", help="Processa só um range e encerra.")
    parser.add_argument("--skip-api-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.token:
        raise SystemExit(
            "Informe o token com --token ou pela variável BTCPUZZLE_USER_TOKEN."
        )

    api = BTCPuzzleAPI(
        user_token=args.token,
        worker_name=args.worker,
        base_url=args.base_url,
    )

    scanner = OpenCLScanner(
        prefer_gpu=not args.cpu,
        platform_hint=args.platform,
        device_hint=args.device,
    )

    if not args.skip_api_test:
        print("Testando conexão com a API...")
        api.test_connection(puzzle_code=38)
        print("API OK.")

    solver = PuzzleSolver(
        api=api,
        scanner=scanner,
        config=SolverConfig(
            puzzle_code=args.puzzle,
            batch_size=args.batch_size,
            ping_interval_seconds=args.ping_interval,
            gpu_count=args.gpu_count,
            once=args.once,
        ),
    )
    solver.install_signal_handlers()
    solver.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
