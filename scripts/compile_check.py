from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Sequence

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compile-check corpus kernels needing profiling")
    parser.add_argument("--kernels", type=Path, default=Path("corpus/kernels"))
    parser.add_argument("--failure-log", type=Path, default=Path("scripts/compile_failures.txt"))
    parser.add_argument("--filter", default="", help="Only compile-check kernel ids containing this substring")
    return parser.parse_args(argv)


def matches_filter(name: str, raw_filter: str) -> bool:
    if not raw_filter:
        return True
    tokens = [token.strip() for token in raw_filter.split(",") if token.strip()]
    return any(token in name for token in tokens)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    compiled = 0
    failed = 0
    failures: list[str] = []
    for kernel_dir in sorted(path for path in args.kernels.iterdir() if path.is_dir()):
        if not matches_filter(kernel_dir.name, args.filter):
            continue
        profile_path = kernel_dir / "profile.json"
        kernel_path = kernel_dir / "kernel.cu"
        if not profile_path.exists() or not kernel_path.exists():
            continue
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        if not profile.get("needs_profiling", False):
            continue
        result = subprocess.run(
            ["nvcc", "-O2", "-arch=sm_80", str(kernel_path), "-o", f"/tmp/gpublind_test_{kernel_dir.name}"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            profile["compile_status"] = "success"
            compiled += 1
        else:
            profile["compile_status"] = "failed"
            failed += 1
            failures.append(f"{kernel_dir.name}: {result.stderr.strip()}")
        profile_path.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    args.failure_log.write_text("\n".join(failures) + ("\n" if failures else ""), encoding="utf-8")
    print(f"Compiled successfully: {compiled}")
    print(f"Compile failed: {failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
