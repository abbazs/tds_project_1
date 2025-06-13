import rich_click as click


@click.group(name="course", help="Embeds the course data")
def cli() -> None:
    pass


if __name__ == "__main__":
    cli()
