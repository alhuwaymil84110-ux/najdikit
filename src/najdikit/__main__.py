"""Allow running NajdiKit via `python3 -m najdikit`."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
