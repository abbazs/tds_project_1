import rich_click as click


@click.group(name="discourse", help="Embeds the Discourse data")
def cli() -> None:
    pass


if __name__ == "__main__":
    cli()
