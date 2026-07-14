"""Compatibility entry point for the VibeWarp web UI.

The former Gradio implementation moved to the Svelte/FastAPI application in
``vibewarp.web``. Keeping this module preserves ``python -m vibewarp.ui``.
"""

from vibewarp.web import create_app, main

__all__ = ["create_app", "main"]

if __name__ == "__main__":
    main()
