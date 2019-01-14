import os
import click

class Cli(click.MultiCommand):
  def __init__(self):
    self.verbose = False
pass_context = click.make_pass_decorator(Cli, ensure=True)

@click.group(chain=True)
@click.option('--verbose', is_flag=True)
@pass_context
def cli(cli_config, verbose):
  click.echo("running cli")
  cli_config.verbos = verbose

@cli.command()
@pass_context
@click.option('-d', '--pdb', default=False, is_flag=True, help="Crash drops into debugger")
@click.option('-r', '--reference', default=False, is_flag=True, help="Test the reference")
@click.option('-f', '--feedback', default=False, is_flag=True, help="Test the feedback")
@click.option('-c', '--control', default=False, is_flag=True, help="Test the controller")
@click.option('-l', '--learning_tracker', default=False, is_flag=True, help="Test the learning tracker")
def test(cli_config, pdb, reference, feedback, control, learning_tracker):
  click.echo("Running tests")

  rv = "python3 -m pytest test/ --ignore=cli -s"
  if pdb:

    rv += " --pdb"

  rv += " -vv"

  if reference:
    rv += " -m reference"

  if feedback:
    rv += " -m feedback"

  if control:
    rv += " -m control"

  if learning_tracker:
    rv += " -m learning_tracker"

  os.system(rv)

