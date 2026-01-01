"""CLI entry point for Semantik plugin tooling."""

from __future__ import annotations


def main() -> None:
    """Run the semantik-plugin CLI."""
    try:
        import click
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise SystemExit("semantik-plugin requires optional deps. Install with 'pip install semantik[cli]'.") from exc

    from semantik_cli.commands import new, validate

    @click.group()
    @click.version_option()
    def cli() -> None:
        """Semantik Plugin Development CLI."""

    cli.add_command(new.new)
    cli.add_command(validate.validate)

    cli()
