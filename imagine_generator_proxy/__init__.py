"""Imagine Generator Proxy package.

Expose `create_app` for WSGI/ASGI or package usage.
"""
from .main import create_app  # noqa: F401

__all__ = ["create_app"]
