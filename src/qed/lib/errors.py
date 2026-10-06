"""One clear line for expected failures; ``QED_DEBUG=1`` restores the traceback."""

from functools import wraps
import os
import sys

import httpx

EXPECTED = (FileNotFoundError, ValueError, RuntimeError, httpx.HTTPError)


def friendly(main):
    @wraps(main)
    def run(*args, **kwargs):
        try:
            return main(*args, **kwargs)
        except EXPECTED as exc:
            if os.environ.get("QED_DEBUG"):
                raise
            print(f"qed: error: {exc}", file=sys.stderr)
            print("(QED_DEBUG=1 shows the full traceback)", file=sys.stderr)
            raise SystemExit(1) from None

    return run
