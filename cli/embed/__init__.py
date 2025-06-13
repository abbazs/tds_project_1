import rich_click as click

from cli.embed import course
from cli.embed import discourse


@click.group(name="embed", help="Embedding cli")
def cli() -> None:
    pass


cli.add_command(course.cli)
cli.add_command(discourse.cli)

if __name__ == "__main__":
    cli()
