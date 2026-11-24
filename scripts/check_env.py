"""Check that the Python environment is ready for this project.

Prints the installed version of every package in requirements.txt (flagging
mismatches), reports which accelerator PyTorch can use (CUDA, Apple MPS or
CPU), and runs a tiny matrix multiplication on it to prove it works.

Usage:
    python scripts/check_env.py
"""

from __future__ import annotations

import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

REQUIREMENTS = Path(__file__).resolve().parent.parent / "requirements.txt"


def read_pins(path: Path) -> dict[str, str]:
    """Return {package: pinned_version} from a requirements file."""
    pins = {}
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if "==" in line:
            name, ver = line.split("==", 1)
            pins[name.strip()] = ver.strip()
    return pins


def check_packages(pins: dict[str, str]) -> bool:
    ok = True
    print("Packages:")
    for name, wanted in pins.items():
        try:
            installed = version(name)
        except PackageNotFoundError:
            print(f"  [MISSING] {name} (want {wanted})")
            ok = False
            continue
        status = "ok" if installed == wanted else f"MISMATCH, want {wanted}"
        if installed != wanted:
            ok = False
        print(f"  [{status:>2}] {name}=={installed}")
    return ok


def check_device() -> bool:
    import torch

    if torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        name = "Apple Metal (MPS)"
    else:
        device = torch.device("cpu")
        name = "CPU only"
    print(f"\nAccelerator: {name} -> torch device '{device}'")

    a = torch.randn(256, 256, device=device)
    result = (a @ a.T).sum().item()
    print(f"Test matmul on {device}: ok (checksum {result:.1f})")
    return device.type != "cpu"


def main() -> int:
    print(f"Python {sys.version.split()[0]} on {platform.platform()}\n")
    packages_ok = check_packages(read_pins(REQUIREMENTS))
    has_accelerator = check_device()

    print()
    if packages_ok and has_accelerator:
        print("Environment ready.")
        return 0
    if not packages_ok:
        print("Some packages are missing or have the wrong version: "
              "run `pip install -r requirements.txt`.")
    if not has_accelerator:
        print("Warning: no GPU found, training would run on the CPU (very slow).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
