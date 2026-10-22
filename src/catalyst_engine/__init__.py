"""Closed-loop catalyst discovery engine."""

__version__ = "0.1.0"

import time as _time

# The app compares this with source-file times to pick up code edits without a server restart (ui/app.py)
LOADED_AT = _time.time()
