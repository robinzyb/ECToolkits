import click

@click.group("cli")
def cli():
    pass

@click.group("wkflow")
def wkflow():
    click.echo('Performing ECToolkits Workflow')