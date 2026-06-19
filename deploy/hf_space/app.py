"""Hugging Face Space entry point — re-exports the FastAPI app from the
`backend` package so the Dockerfile's `uvicorn app:app` finds it.
"""

from backend.api import app  # noqa: F401
