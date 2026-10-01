import argparse
import os
from aiohttp import web
from .main import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="NovelAI image generator proxy")
    parser.add_argument("--port", type=int, default=None,
        help="port to bind (default: PORT env var, or 8080)")
    parser.add_argument("--host", default="0.0.0.0",
        help="interface to bind (default: 0.0.0.0)")
    args = parser.parse_args()

    port = args.port if args.port is not None else int(os.environ.get("PORT", "8080"))
    web.run_app(create_app(), host=args.host, port=port)


if __name__ == "__main__":
    main()
