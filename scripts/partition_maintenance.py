#!/usr/bin/env python3
"""Partition maintenance script for DBA operations.

This script provides utilities for monitoring and maintaining the chunk table partitions
in the Semantik database. It includes health checks, monitoring reports, and
recommendations for partition rebalancing.
"""

import argparse
import asyncio
import sys
from datetime import UTC, datetime
from typing import Any

import asyncpg
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from tabulate import tabulate  # type: ignore[import-untyped]

# Add the parent directory to the path to import shared modules
sys.path.insert(0, "/app")

from shared.database.database import get_database_url

console = Console()


class PartitionMonitor:
    """Monitor and analyze chunk table partitions."""

    def __init__(self, db_url: str):
        self.db_url = db_url

    async def connect(self) -> asyncpg.Connection:
        """Create a database connection."""
        return await asyncpg.connect(self.db_url)

    async def get_partition_health_summary(self) -> list[dict[str, Any]]:
        """Get partition health summary from the monitoring view."""
        conn = await self.connect()
        try:
            rows = await conn.fetch("SELECT * FROM partition_health_summary ORDER BY partition_num")
            return [dict(row) for row in rows]
        finally:
            await conn.close()

    async def get_partition_distribution(self) -> list[dict[str, Any]]:
        """Get chunk distribution across partitions."""
        conn = await self.connect()
        try:
            rows = await conn.fetch("SELECT * FROM partition_chunk_distribution ORDER BY partition_num")
            return [dict(row) for row in rows]
        finally:
            await conn.close()

    async def get_partition_sizes(self) -> list[dict[str, Any]]:
        """Get partition size information."""
        conn = await self.connect()
        try:
            rows = await conn.fetch("SELECT * FROM partition_size_distribution ORDER BY partition_num")
            return [dict(row) for row in rows]
        finally:
            await conn.close()

    async def get_hot_partitions(self) -> list[dict[str, Any]]:
        """Get hot partition information."""
        conn = await self.connect()
        try:
            rows = await conn.fetch("SELECT * FROM partition_hot_spots ORDER BY chunks_last_hour DESC")
            return [dict(row) for row in rows]
        finally:
            await conn.close()

    async def analyze_skew(self) -> list[dict[str, Any]]:
        """Analyze partition skew using the stored function."""
        conn = await self.connect()
        try:
            rows = await conn.fetch("SELECT * FROM analyze_partition_skew()")
            return [dict(row) for row in rows]
        finally:
            await conn.close()

    async def refresh_materialized_view(self) -> None:
        """Manually refresh the collection_chunking_stats materialized view."""
        conn = await self.connect()
        try:
            await conn.execute("SELECT refresh_collection_chunking_stats()")
            console.print("[green]Successfully refreshed collection_chunking_stats materialized view[/green]")
        finally:
            await conn.close()

    async def get_collection_stats(self) -> list[dict[str, Any]]:
        """Get collection chunking statistics."""
        conn = await self.connect()
        try:
            rows = await conn.fetch("SELECT * FROM collection_chunking_stats ORDER BY total_chunks DESC")
            return [dict(row) for row in rows]
        finally:
            await conn.close()

    def print_health_report(self, health_data: list[dict[str, Any]]) -> None:
        """Print a formatted health report."""
        table = Table(title="Partition Health Summary", show_lines=True)
        table.add_column("Partition", style="cyan", no_wrap=True)
        table.add_column("Chunks", justify="right")
        table.add_column("Size", justify="right")
        table.add_column("Chunk %", justify="right")
        table.add_column("Size %", justify="right")
        table.add_column("Status", justify="center")
        table.add_column("Recommendation", style="dim")

        for row in health_data:
            status_color = {"HEALTHY": "green", "WARNING": "yellow", "UNBALANCED": "red"}.get(
                row["health_status"], "white"
            )

            table.add_row(
                str(row["partition_num"]),
                str(row["chunk_count"]),
                row["partition_size"],
                f"{row['chunk_percentage']:.1f}%",
                f"{row['size_percentage']:.1f}%",
                Text(row["health_status"], style=status_color),
                row["recommendation"],
            )

        console.print(table)

    def print_skew_analysis(self, skew_data: list[dict[str, Any]]) -> None:
        """Print skew analysis results."""
        panel_content = ""
        for metric in skew_data:
            status_color = {
                "HEALTHY": "green",
                "INFO": "blue",
                "WARNING": "yellow",
                "MODERATE": "yellow",
                "HIGH": "red",
                "CRITICAL": "red",
            }.get(metric["status"], "white")

            panel_content += f"[bold]{metric['metric']}:[/bold] "
            panel_content += f"[{status_color}]{metric['value']:.2f} ({metric['status']})[/{status_color}]\n"
            panel_content += f"  {metric['details']}\n\n"

        console.print(Panel(panel_content.strip(), title="Partition Skew Analysis", border_style="blue"))

    def print_hot_partitions(self, hot_data: list[dict[str, Any]]) -> None:
        """Print hot partition information."""
        if not hot_data:
            console.print("[green]No hot partitions detected[/green]")
            return

        table = Table(title="Hot Partitions (High Activity)", show_lines=True)
        table.add_column("Partition", style="cyan", no_wrap=True)
        table.add_column("Last Hour", justify="right")
        table.add_column("Last Day", justify="right")
        table.add_column("Last Week", justify="right")
        table.add_column("Hour %", justify="right")
        table.add_column("Day %", justify="right")

        for row in hot_data[:5]:  # Show top 5 hot partitions
            hour_color = (
                "red" if row["hour_percentage"] > 12.5 else "yellow" if row["hour_percentage"] > 6.25 else "green"
            )
            day_color = "red" if row["day_percentage"] > 12.5 else "yellow" if row["day_percentage"] > 6.25 else "green"

            table.add_row(
                str(row["partition_num"]),
                str(row["chunks_last_hour"]),
                str(row["chunks_last_day"]),
                str(row["chunks_last_week"]),
                Text(f"{row['hour_percentage']:.1f}%", style=hour_color),
                Text(f"{row['day_percentage']:.1f}%", style=day_color),
            )

        console.print(table)

    def generate_rebalancing_recommendations(self, health_data: list[dict[str, Any]]) -> list[str]:
        """Generate specific rebalancing recommendations based on partition health."""
        recommendations = []

        # Find unbalanced partitions
        unbalanced = [p for p in health_data if p["health_status"] == "UNBALANCED"]
        warning = [p for p in health_data if p["health_status"] == "WARNING"]

        if unbalanced:
            recommendations.append(f"CRITICAL: {len(unbalanced)} partition(s) are significantly unbalanced")
            for p in unbalanced:
                recommendations.append(f"  - Partition {p['partition_num']}: {p['chunk_percentage']:.1f}% of chunks")
            recommendations.append("\nRecommended Actions:")
            recommendations.append("1. Consider adjusting the partition key hash function")
            recommendations.append("2. Review document distribution patterns")
            recommendations.append("3. Implement partition rebalancing during low-traffic periods")

        if warning:
            recommendations.append(f"\nWARNING: {len(warning)} partition(s) showing signs of imbalance")
            recommendations.append("Monitor these partitions closely and plan preventive maintenance")

        if not unbalanced and not warning:
            recommendations.append("All partitions are healthy and well-balanced")

        return recommendations


async def main() -> None:
    """Main entry point for the partition maintenance script."""
    parser = argparse.ArgumentParser(description="Partition maintenance and monitoring tool")
    parser.add_argument(
        "command",
        choices=["health", "distribution", "sizes", "hot", "skew", "refresh", "collections", "full"],
        help="Command to execute",
    )
    parser.add_argument("--db-url", help="Database URL (overrides environment variable)")
    parser.add_argument("--format", choices=["table", "json"], default="table", help="Output format")

    args = parser.parse_args()

    # Get database URL
    db_url = args.db_url or get_database_url()
    if not db_url:
        console.print("[red]Error: Database URL not provided and not found in environment[/red]")
        sys.exit(1)

    monitor = PartitionMonitor(db_url)

    try:
        if args.command == "health":
            health_data = await monitor.get_partition_health_summary()
            monitor.print_health_report(health_data)

            # Generate recommendations
            recommendations = monitor.generate_rebalancing_recommendations(health_data)
            if recommendations:
                console.print("\n[bold]Recommendations:[/bold]")
                for rec in recommendations:
                    console.print(rec)

        elif args.command == "distribution":
            dist_data = await monitor.get_partition_distribution()
            headers = ["Partition", "Chunks", "Documents", "Collections", "Avg Tokens", "Chunk %"]
            rows = [
                [
                    row["partition_num"],
                    row["chunk_count"],
                    row["document_count"],
                    row["collection_count"],
                    f"{row['avg_tokens']:.1f}",
                    f"{row['chunk_percentage']:.1f}%",
                ]
                for row in dist_data
            ]
            console.print(tabulate(rows, headers=headers, tablefmt="grid"))

        elif args.command == "sizes":
            size_data = await monitor.get_partition_sizes()
            headers = ["Partition", "Size", "Size %", "Row Estimate"]
            rows = [
                [row["partition_num"], row["size_pretty"], f"{row['size_percentage']:.1f}%", row["row_estimate"]]
                for row in size_data
            ]
            console.print(tabulate(rows, headers=headers, tablefmt="grid"))

        elif args.command == "hot":
            hot_data = await monitor.get_hot_partitions()
            monitor.print_hot_partitions(hot_data)

        elif args.command == "skew":
            skew_data = await monitor.analyze_skew()
            monitor.print_skew_analysis(skew_data)

        elif args.command == "refresh":
            await monitor.refresh_materialized_view()

        elif args.command == "collections":
            stats = await monitor.get_collection_stats()
            headers = ["Collection", "Chunked Docs", "Total Chunks", "Avg Tokens", "Last Created"]
            rows = [
                [
                    row["name"],
                    row["chunked_documents"],
                    row["total_chunks"],
                    f"{row['avg_tokens_per_chunk']:.1f}" if row["avg_tokens_per_chunk"] else "N/A",
                    row["last_chunk_created"].strftime("%Y-%m-%d %H:%M") if row["last_chunk_created"] else "Never",
                ]
                for row in stats
            ]
            console.print(tabulate(rows, headers=headers, tablefmt="grid"))

        elif args.command == "full":
            # Full report
            console.print("[bold]=== Partition Maintenance Report ===[/bold]\n")
            console.print(f"Generated at: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')}\n")

            # Health summary
            health_data = await monitor.get_partition_health_summary()
            monitor.print_health_report(health_data)

            # Hot partitions
            console.print("\n")
            hot_data = await monitor.get_hot_partitions()
            monitor.print_hot_partitions(hot_data)

            # Skew analysis
            console.print("\n")
            skew_data = await monitor.analyze_skew()
            monitor.print_skew_analysis(skew_data)

            # Recommendations
            console.print("\n")
            recommendations = monitor.generate_rebalancing_recommendations(health_data)
            if recommendations:
                console.print("[bold]Recommendations:[/bold]")
                for rec in recommendations:
                    console.print(rec)

    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
