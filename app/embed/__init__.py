import rich_click as click

from app.embed import course
from app.embed import discourse


@click.group(name="embed", help="Embedding cli")
def cli() -> None:
    pass


cli.add_command(course.embed, name="course")
cli.add_command(discourse.embed, name="discourse")

if __name__ == "__main__":
    cli()
