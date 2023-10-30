import click
import yaml
from ectoolkits.workflows.calc_diel import calc_diel

# --- Helper functions ---
def yaml_to_dict(yaml_file: str):
    """
    Convert a yaml file to a dictionary
    """
    with open(yaml_file, 'r') as fr:
        input_dict = yaml.safe_load(fr)
    return input_dict

def batch_yaml_to_dict(input: str, 
                       machine: str, 
                       resources: str):
    """
    Convert a yaml file to a dictionary
    """
    input_dict = yaml_to_dict(input)
    machine_dict = yaml_to_dict(machine)
    resources_dict = yaml_to_dict(resources)
    return input_dict, machine_dict, resources_dict
# --- End of helper functions ---

@click.group("cli")
def cli():
    pass

@cli.group("wkflow")
def wkflow():
    click.echo('Performing ECToolkits Workflow')

## --input_file put every into a yaml file! keep it simple
## --machine
## --resources
@wkflow.command("calc_diel")
@click.option('--input', '-i', 
              type=click.Path(exists=True), help='Input file for workflow')
@click.option('--machine', '-m', 
              type=click.Path(exists=True), help='Machine to run the workflow on')
@click.option('--resources', '-r',
              type=click.Path(exists=True), help='Resources to use for the workflow')
def calc_diel_cli(input, machine, resources):
    input_dict, machine_dict, resources_dict = \
        batch_yaml_to_dict(input, machine, resources)
    scale = input_dict.pop("scale")
    if scale == 'global':
        calc_diel(**input_dict, 
                machine_dict=machine_dict, 
                resources_dict=resources_dict)
    elif scale == 'atomic':
        pass
    else:
        print(f"The scale {scale} is not supported!")
# change the calc_diel function