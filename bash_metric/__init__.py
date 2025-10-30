# -*- coding: utf-8 -*-
"""Nsight Compute/Systems profiling helpers exposed as a Python package."""

from __future__ import annotations

from importlib.metadata import version, PackageNotFoundError

__all__ = ["__version__"]

try:
    __version__ = version("bash-metric")
except PackageNotFoundError:
    # When running from source before packaging, fall back to dev tag.
    __version__ = "0.0.dev0"

