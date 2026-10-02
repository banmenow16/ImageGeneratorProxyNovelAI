import argparse
import os
from aiohttp import web
from .server.app import configure_logging, create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="NovelAI image generator proxy")
    parser.add_argument("--port", type=int, default=None,
        help="port to bind (default: PORT env var, or 8080)")
    parser.add_argument("--host", default="0.0.0.0",
        help="interface to bind (default: 0.0.0.0)")
    args = parser.parse_args()

    configure_logging()
    port = args.port if args.port is not None else int(os.environ.get("PORT", "8080"))
    # access_log=None: the app's request_logging_middleware covers request
    # logging (incl. slow-request and unhandled-exception context), so
    # aiohttp's built-in access logger would just duplicate every line.
    web.run_app(create_app(), host=args.host, port=port, access_log=None)


if __name__ == "__main__":
    main()
