"""CLI entry point for the Semantik MCP server."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

from webui.mcp.server import SemantikMCPServer


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="semantik-mcp", description="Semantik MCP server (stdio transport)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="Start the MCP server (stdio transport)")
    serve.add_argument(
        "--profile",
        "-p",
        action="append",
        dest="profiles",
        default=None,
        help="Profile(s) to expose. Can be provided multiple times. If omitted, all enabled profiles are exposed.",
    )
    serve.add_argument(
        "--webui-url",
        default=os.getenv("SEMANTIK_WEBUI_URL", "http://localhost:8080"),
        help="Semantik WebUI base URL (or SEMANTIK_WEBUI_URL).",
    )
    serve.add_argument(
        "--auth-token",
        default=os.getenv("SEMANTIK_AUTH_TOKEN"),
        help="Semantik auth token (JWT access token or API key) (or SEMANTIK_AUTH_TOKEN).",
    )
    serve.add_argument(
        "--log-level",
        default=os.getenv("SEMANTIK_MCP_LOG_LEVEL", "INFO"),
        help="Logging level (or SEMANTIK_MCP_LOG_LEVEL).",
    )
    serve.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output (sets log level to DEBUG, overrides --log-level).",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command != "serve":
        raise SystemExit(f"Unknown command: {args.command}")

    if not args.auth_token:
        parser.error("Missing --auth-token (or SEMANTIK_AUTH_TOKEN)")

    # --verbose overrides --log-level
    log_level = "DEBUG" if args.verbose else str(args.log_level).upper()
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    server = SemantikMCPServer(
        webui_url=args.webui_url,
        auth_token=args.auth_token,
        profile_filter=args.profiles,
    )

    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except Exception as exc:
        logging.getLogger(__name__).error("MCP server failed: %s", exc, exc_info=True)
        raise SystemExit(1) from exc


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
