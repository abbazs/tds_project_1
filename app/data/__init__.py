import rich_click as click

from app.data import course
from app.data import discourse


@click.group(name="scrape", help="Scrapping cli")
def cli() -> None:
    pass


cli.add_command(course.cli)
cli.add_command(discourse.cli)

if __name__ == "__main__":
    cli()