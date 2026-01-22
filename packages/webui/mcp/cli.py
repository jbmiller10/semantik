"""CLI entry point for the Semantik MCP server."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

from webui.mcp.server import SemantikMCPServer


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="semantik-mcp", description="Semantik MCP server")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="Start the MCP server")
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
        help="Semantik auth token (JWT access token or API key) (or SEMANTIK_AUTH_TOKEN). Required for stdio transport.",
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
    # HTTP transport options
    serve.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="Transport type: 'stdio' for local process communication, 'http' for remote HTTP access.",
    )
    serve.add_argument(
        "--http-host",
        default=os.getenv("SEMANTIK_MCP_HTTP_HOST", "0.0.0.0"),
        help="HTTP server bind host (or SEMANTIK_MCP_HTTP_HOST). Only used with --transport http.",
    )
    serve.add_argument(
        "--http-port",
        type=int,
        default=int(os.getenv("SEMANTIK_MCP_HTTP_PORT", "9090")),
        help="HTTP server port (or SEMANTIK_MCP_HTTP_PORT). Only used with --transport http.",
    )

    return parser


def _get_internal_api_key() -> str | None:
    """Load internal API key from shared file (for HTTP/service mode)."""
    try:
        from shared.config import settings
        from shared.config.internal_api_key import ensure_internal_api_key

        result: str | None = ensure_internal_api_key(settings)
        return result
    except Exception as exc:
        logging.getLogger(__name__).warning("Failed to load internal API key: %s", exc)
        return None


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command != "serve":
        raise SystemExit(f"Unknown command: {args.command}")

    # --verbose overrides --log-level
    log_level = "DEBUG" if args.verbose else str(args.log_level).upper()
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger(__name__)

    if args.transport == "http":
        # HTTP transport: use internal API key for service mode
        internal_key = _get_internal_api_key()
        if not internal_key:
            parser.error(
                "Failed to load internal API key for HTTP transport. "
                "Ensure the WebUI has been started at least once to generate the key, "
                "or set INTERNAL_API_KEY environment variable."
            )

        logger.info("HTTP transport: using internal API key for service mode")
        server = SemantikMCPServer(
            webui_url=args.webui_url,
            internal_api_key=internal_key,
            profile_filter=args.profiles,
        )
    else:
        # stdio transport: requires user auth token
        if not args.auth_token:
            parser.error("Missing --auth-token (or SEMANTIK_AUTH_TOKEN) for stdio transport")

        server = SemantikMCPServer(
            webui_url=args.webui_url,
            auth_token=args.auth_token,
            profile_filter=args.profiles,
        )

    try:
        if args.transport == "http":
            asyncio.run(server.run_http(host=args.http_host, port=args.http_port))
        else:
            asyncio.run(server.run())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except Exception as exc:
        logger.error("MCP server failed: %s", exc, exc_info=True)
        raise SystemExit(1) from exc


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
