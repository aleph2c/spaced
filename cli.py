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
def test(cli_config, pdb):
  click.echo("Running tests")
  if pdb:
    rv = "python3 -m pytest test/ --ignore=cli --pdb -s"
  else:
    rv = "python3 -m pytest test/ --ignore=cli -s"
  os.system(rv)
