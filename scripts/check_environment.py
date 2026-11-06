"""Report whether a model-specific SuJevi environment meets its declared floor.

This script does not download weights or load a model. Run it with the Python
executable belonging to the environment being checked.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import sys
from importlib import metadata
from typing import Any


KEV_REQUIREMENTS = {
    "accelerate": ">=1.15.0",
    "datasets": ">=3.0",
    "numpy": ">=2.5.3",
    "peft": ">=0.21",
    "pydantic": ">=2.9",
    "scikit-learn": ">=1.9.1",
    "torch": ">=2.6,<2.9",
    "transformers": ">=5.17,<6",
    "fastapi": ">=0.115",
    "typesafe-sdk": ">=0.6.0",
    "uvicorn": ">=0.30",
}

JEFF_REQUIREMENTS = {
    "torch": "==2.13.0",
    "transformers": "==5.17.0",
    "safetensors": "==0.8.0",
    "huggingface_hub": "==1.32.0",
    "accelerate": "==1.15.0",
    "jeff": "==1.3.0",
}

LAYA_REQUIREMENTS = {
    "torch": "==2.13.0",
    "transformers": "==5.17.0",
    "safetensors": "==0.8.0",
    "huggingface_hub": "==1.32.0",
    "numpy": "==2.5.3",
    "laya": "==0.4.24",
    "vegaml": ">=0.15.0",
}

BOSUN_REQUIREMENTS = {
    "torch": "==2.13.0",
    "transformers": "==4.57.0",
    "peft": "==0.21.0",
    "safetensors": "==0.8.0",
    "huggingface_hub": "==0.36.0",
    "accelerate": "==1.15.0",
    "numpy": "==2.5.3",
}


def package_report(requirements: dict[str, str]) -> tuple[dict[str, Any], list[str]]:
    try:
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version
    except ImportError:
        return {}, ["packaging is missing; the environment cannot validate versions"]

    report: dict[str, Any] = {}
    failures: list[str] = []
    for distribution, constraint in requirements.items():
        try:
            installed = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            report[distribution] = {"required": constraint, "installed": None, "ok": False}
            failures.append(f"missing package: {distribution}{constraint}")
            continue

        ok = Version(installed) in SpecifierSet(constraint)
        report[distribution] = {"required": constraint, "installed": installed, "ok": ok}
        if not ok:
            failures.append(f"version mismatch: {distribution} {installed} does not satisfy {constraint}")
    return report, failures


def cuda_report() -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    try:
        import torch
    except ImportError:
        return {"available": False, "error": "torch import failed"}, ["torch import failed"]

    available = torch.cuda.is_available()
    report: dict[str, Any] = {
        "available": available,
        "torch_cuda_runtime": torch.version.cuda,
        "device_count": torch.cuda.device_count() if available else 0,
    }
    if not available:
        failures.append("CUDA is not available to PyTorch")
        return report, failures

    properties = torch.cuda.get_device_properties(0)
    report.update(
        {
            "device_0": properties.name,
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "total_memory_gib": round(properties.total_memory / 1024**3, 2),
        }
    )
    return report, failures


def c_compiler_report() -> tuple[dict[str, Any], list[str]]:
    """Locate the C compiler Triton needs to build its runtime helper."""
    configured = os.environ.get("CC")
    candidates = [configured] if configured else []
    candidates.extend(["cc", "gcc", "x86_64-conda-linux-gnu-cc"])
    for candidate in candidates:
        if candidate and (executable := shutil.which(candidate)):
            return {
                "available": True,
                "configured_by_cc": configured,
                "executable": executable,
            }, []
    return {
        "available": False,
        "configured_by_cc": configured,
        "executable": None,
    }, ["C compiler is missing; Triton cannot compile its CUDA runtime helper"]


def check_kev() -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    python_ok = (3, 12) <= sys.version_info[:2] < (3, 14)
    if not python_ok:
        failures.append("Kev requires Python >=3.12,<3.14")

    packages, package_failures = package_report(KEV_REQUIREMENTS)
    failures.extend(package_failures)

    kev_importable = importlib.util.find_spec("kev") is not None
    if not kev_importable:
        failures.append("the official kev package is not importable")

    cuda, cuda_failures = cuda_report()
    failures.extend(cuda_failures)

    return (
        {
            "profile": "kev",
            "python": {
                "version": platform.python_version(),
                "executable": sys.executable,
                "required": ">=3.12,<3.14",
                "ok": python_ok,
            },
            "platform": platform.platform(),
            "kev_importable": kev_importable,
            "packages": packages,
            "cuda": cuda,
        },
        failures,
    )


def check_jeff() -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    python_ok = (3, 12) <= sys.version_info[:2] < (3, 13)
    if not python_ok:
        failures.append("Jeff profile requires Python >=3.12,<3.13")

    packages, package_failures = package_report(JEFF_REQUIREMENTS)
    failures.extend(package_failures)

    jeff_importable = importlib.util.find_spec("jeff") is not None
    if not jeff_importable:
        failures.append("the official jeff package is not importable")

    cuda, cuda_failures = cuda_report()
    failures.extend(cuda_failures)

    c_compiler, compiler_failures = c_compiler_report()
    failures.extend(compiler_failures)

    return (
        {
            "profile": "jeff",
            "python": {
                "version": platform.python_version(),
                "executable": sys.executable,
                "required": ">=3.12,<3.13",
                "ok": python_ok,
            },
            "platform": platform.platform(),
            "jeff_importable": jeff_importable,
            "packages": packages,
            "c_compiler": c_compiler,
            "cuda": cuda,
        },
        failures,
    )


def check_laya() -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    python_ok = (3, 12) <= sys.version_info[:2] < (3, 13)
    if not python_ok:
        failures.append("Laya profile requires Python >=3.12,<3.13")

    packages, package_failures = package_report(LAYA_REQUIREMENTS)
    failures.extend(package_failures)

    laya_importable = importlib.util.find_spec("laya") is not None
    if not laya_importable:
        failures.append("the official laya package is not importable")

    cuda, cuda_failures = cuda_report()
    failures.extend(cuda_failures)

    return (
        {
            "profile": "laya",
            "python": {
                "version": platform.python_version(),
                "executable": sys.executable,
                "required": ">=3.12,<3.13",
                "ok": python_ok,
            },
            "platform": platform.platform(),
            "laya_importable": laya_importable,
            "packages": packages,
            "cuda": cuda,
        },
        failures,
    )


def check_bosun() -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    python_ok = (3, 12) <= sys.version_info[:2] < (3, 13)
    if not python_ok:
        failures.append("Bosun profile requires Python >=3.12,<3.13")

    packages, package_failures = package_report(BOSUN_REQUIREMENTS)
    failures.extend(package_failures)

    cuda, cuda_failures = cuda_report()
    failures.extend(cuda_failures)

    return (
        {
            "profile": "bosun",
            "python": {
                "version": platform.python_version(),
                "executable": sys.executable,
                "required": ">=3.12,<3.13",
                "ok": python_ok,
            },
            "platform": platform.platform(),
            "packages": packages,
            "cuda": cuda,
        },
        failures,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("kev", "jeff", "laya", "bosun"), required=True)
    args = parser.parse_args()

    if args.profile == "kev":
        report, failures = check_kev()
    elif args.profile == "jeff":
        report, failures = check_jeff()
    elif args.profile == "laya":
        report, failures = check_laya()
    elif args.profile == "bosun":
        report, failures = check_bosun()
    else:  # pragma: no cover - argparse prevents this branch
        raise AssertionError(args.profile)

    report["ok"] = not failures
    report["failures"] = failures
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
