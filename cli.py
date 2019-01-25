"""cli

The cli module provides a command line user interface to the ``spaced`` package.
It is mostly a wrapper which sets a number of aliases for long py.test
commands, which can be used to test different parts of the ``spaced`` package.
The cli module is heavily reliant upon `click <https://click.palletsprojects.com/en/7.x/>`_.

**Example(s)**:
  Here are the commands that are available from a locally installed spaced
  package.

  .. code-block:: bash

    # to install spaced locally and create the command entry points
    $ cd <spaced directory>
    $ pip install --editable .

    # to test everything
    $ space test

    # to test the reference features
    $ space test -r

    # to test the feedback features
    $ space test -f

    # to test the controller features
    $ space test -c

"""
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

