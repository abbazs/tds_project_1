"""aws/utils/__init__.py - Shared utility functions for AWS CLI tool."""

import sys
from logging import INFO
from logging import Formatter
from logging import Logger
from logging import StreamHandler
from logging import getLogger
from typing import Any

from rich.console import Console
from rich.table import Table

console = Console()

STATUS_EMOJI = {
    "running": "✅",
    "stopped": "⏹️",
    "pending": "⏳",
    "stopping": "🛑",
    "terminated": "💀",
    "shutting-down": "🔄",
    "healthy": "💚",
    "unhealthy": "❤️",
    "warning": "⚠️",
    "error": "🔥",
    "success": "🎉",
    "failure": "❌",
    "unknown": "❓",
    "important": "💡",
}


def print_table(
    title: str,
    columns: list[str],
    rows: list[list[Any]],
    column_styles: dict[str, str] | None = None,
    status_column: int | None = None,
) -> None:
    """Print a formatted table using Rich."""
    table = Table(title=title, show_header=True, header_style="bold")
    for col in columns:
        table.add_column(
            col,
            style=column_styles.get(col) if column_styles else None,
            no_wrap=True,
            overflow="ignore",
        )
    for row in rows:
        formatted_row = [str(cell or "") for cell in row]
        if status_column is not None and row[status_column].lower() in STATUS_EMOJI:
            formatted_row[status_column] = (
                f"{STATUS_EMOJI[row[status_column].lower()]} {formatted_row[status_column]}"
            )
        table.add_row(*formatted_row)
    console.print(table)


def print_success(message: str) -> None:
    console.print(f"{STATUS_EMOJI['success']}  {message}", style="bold green")


def print_warning(message: str) -> None:
    console.print(f"{STATUS_EMOJI['warning']}  {message}", style="bold yellow")


def print_error(message: str) -> None:
    console.print(f"{STATUS_EMOJI['error']}  {message}", style="bold red")


def print_important(message: str) -> None:
    console.print(f"{STATUS_EMOJI['important']}  {message}", style="bold blue")


def error_exit(message: str, exit_code: int = 1) -> None:
    print_error(message)
    sys.exit(exit_code)


def setup_logger(name: str) -> Logger:
    logger = getLogger(name)
    if not logger.handlers:
        handler = StreamHandler()
        handler.setFormatter(
            Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(INFO)
    return logger
