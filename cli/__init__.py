from pathlib import Path

import rich_click as click
import toml
from rich.console import Console
from rich.prompt import Confirm

from cli import embed
from cli import data
from cli.utils import error_exit
from cli.utils import print_success
from cli.utils import print_table
from cli.utils import print_warning

console = Console()


@click.group(
    name="cli",
    help="TDS Project 1 Data Prepration Tool - command-line interface",
)
def cli() -> None:
    pass


@cli.command(
    "self-update", help="Auto-update [project.scripts] in pyproject.toml"
)
def self_update() -> None:
    root_dir = Path(__file__).resolve().parent.parent
    cli_dir = root_dir / "cli"
    toml_path = root_dir / "pyproject.toml"

    if not toml_path.exists():
        error_exit("pyproject.toml not found at project root.")

    def discover_scripts() -> dict[str, str]:
        entries = {}
        for path in cli_dir.rglob("*.py"):
            if not path.is_file():
                continue
            source = path.read_text(encoding="utf-8")
            lines = [
                line.strip()
                for line in source.strip().splitlines()
                if line.strip()
            ]

            if not (
                len(lines) >= 2
                and lines[-2] == 'if __name__ == "__main__":'
                and "cli()" in lines[-1]
            ):
                continue

            if path.name == "__init__.py":
                rel_path = path.relative_to(cli_dir).parent
                module_parts = rel_path.parts
                if module_parts:
                    script_name = module_parts[-1]
                    fqdn = f"cli.{'.'.join(module_parts)}:cli"
                    entries[script_name] = fqdn
                else:
                    entries["cli"] = "cli:cli"
        return entries

    new_entries = discover_scripts()

    with toml_path.open("r", encoding="utf-8") as f:
        data = toml.load(f)

    project = data.get("project", {})
    existing = project.get("scripts", {})
    added = {k: v for k, v in new_entries.items() if k not in existing}
    removed = {k: v for k, v in existing.items() if k not in new_entries}
    unchanged = {k: v for k, v in existing.items() if k in new_entries}

    # Build rows for table output
    all_rows = [[k, v, "existing"] for k, v in unchanged.items()]

    if added or removed:
        all_rows.append(["", "", ""])  # visual separator

    for k, v in added.items():
        all_rows.append([k, v, "new"])

    for k, v in removed.items():
        all_rows.append([k, v, "removed"])

    column_styles = {"Script": "cyan", "Entry Point": "green", "Status": "dim"}
    print_table(
        title="[project.scripts]",
        columns=["Script", "Entry Point", "Status"],
        rows=all_rows,
        column_styles=column_styles,
        status_column=2,
    )

    if not added and not removed:
        console.print("No changes detected.", style="dim")
        return

    if Confirm.ask("Do you want to apply these changes to pyproject.toml?"):
        updated_scripts = {**unchanged, **added}
        project["scripts"] = updated_scripts
        data["project"] = project

        with toml_path.open("w", encoding="utf-8") as f:
            toml.dump(data, f)

        for k in added:
            print_success(f"Added script: {k}")
        for k in removed:
            print_warning(f"Removed script: {k}")
    else:
        print_warning("Aborted. No changes made.")


cli.add_command(embed.cli)
cli.add_command(data.cli)


if __name__ == "__main__":
    cli()
