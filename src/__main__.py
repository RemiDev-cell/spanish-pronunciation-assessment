"""Allow `python -m src` as an alias for the CLI."""

from src.pipeline import main

if __name__ == "__main__":
    raise SystemExit(main())
