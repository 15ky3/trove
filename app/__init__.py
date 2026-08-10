"""Trove — a local model library for the Hugging Face Hub."""

import os

# docker compose passes variables it cannot resolve as an empty string.
# huggingface_hub reads HF_ENDPOINT into a constant at import time and would
# take an empty value literally (=> "Request URL is missing a protocol").
# This has to run before any huggingface_hub import, hence its place up here.
for _var in ("HF_ENDPOINT", "HF_TOKEN", "UI_PASSWORD", "HF_HOME"):
    if _var in os.environ and not os.environ[_var].strip():
        del os.environ[_var]

__version__ = "1.2"
