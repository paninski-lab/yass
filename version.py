"""
Script for creating new releases
"""
import ast
import re
from subprocess import call as _call

import click

TESTING = True


def replace_in_file(path_to_file, original, replacement):
    """Replace string in file
    """
    with open(path_to_file, 'r+') as f:
        content = f.read()
        content.replace(original, replacement)
        f.seek(0)
        f.write(content)
        f.truncate()


def call(*args, **kwargs):
    """Mocks call function for testing
    """
    if TESTING:
        print(args, kwargs)
        return 0
    else:
        return _call(*args, **kwargs)


class Versioner(object):
    """Utility functions to manage versions
    """

    @classmethod
    def current_version(cls):
        """Returns the current version in __init__.py
        """
        _version_re = re.compile(r'__version__\s+=\s+(.*)')

        with open('src/yass/__init__.py', 'rb') as f:
            VERSION = str(ast.literal_eval(_version_re.search(
                            f.read().decode('utf-8')).group(1)))

        return VERSION

    @classmethod
    def release_version(cls):
        """
        Returns a release version number
        e.g. 2.4.4dev -> v.2.2.4
        """
        current = cls.current_version()

        if 'dev' not in current:
            raise ValueError('Current version is not a dev version')

        return current.replace('dev', '')

    @classmethod
    def bump_up_version(cls):
        """
        Gets gets a release version and returns a the next value value.
        e.g. v1-2-5 -> v1-2-6dev
        """
        # Get current version
        current = cls.current_version()

        if 'dev' in current:
            raise ValueError('Current version is dev version, new dev '
                             'versions can only be made from release versions')

        # Get Z from vX-Y-Z and sum 1
        new_subversion = (int(re.search(r'v\d+-\d+-(\d+)', current).group(1)) +
                          1)

        # Replace new_subversion in current version
        new_version = re.sub(r'(v\d+-\d+-)(\d+)', r'\g<1>{}dev'
                             .format(new_subversion), current)

        return new_version

    @classmethod
    def save_version(cls, new_version, message='', tag=False):
        """
        Replaces version in  app.yaml and optionally creates a tag in the git
        repository (also saves a commit)
        """
        current = cls.current_version()

        # Save version in app.yaml
        replace_in_file('app.yaml', {current: new_version})

        # Create tag
        if tag:
            # Run git add and git status
            click.echo('Adding new changes to the repository...')
            _call(['git', 'add', '--all'])
            _call(['git', 'status'])

            # Commit repo with updated dev version
            click.echo('Creating new commit release version...')
            msg = 'Release {}'.format(new_version)
            _call(['git', 'commit', '-m', msg])

            tag_name = new_version.replace('-', '.')
            click.echo('Creating tag {}...'.format(tag_name))
            _call(['git', 'tag', '-a', tag_name, '-m', message])

            click.echo('Pushing tags...')
            _call(['git', 'push', '--tags'])


@click.group()
def cli():
    pass


@cli.command(help='Sets a new version for the project: saves the new version '
                  'in the app.yaml version key, creates a tag with the '
                  'version number and bumps up version')
def version():
    current = Versioner.current_version()
    release = Versioner.release_version()

    click.confirm('Current version in app.yaml is {current} release version '
                  'will be {release}, do you want to continue?'
                  .format(current=current, release=release), abort=True)

    click.confirm('Before continuing modify the CHANGELOG file. Continue '
                  'done', abort=True)

    # Ask for a message to include in the tag
    message = click.prompt('What is this release about? ')

    # Replace version number and create tag
    Versioner.save_version(release, message, tag=True)

    # Create a new dev version and save it
    bumped_version = Versioner.bump_up_version()
    click.echo('Setting new dev version to: {}'.format(bumped_version))
    Versioner.save_version(bumped_version)

    # Run git add and git status
    click.echo('Adding new changes to the repository...')
    call(['git', 'add', '--all'])
    call(['git', 'status'])

    # Commit repo with updated dev version
    click.echo('Creating new commit with new dev version...')
    msg = 'Bumps up project to version {}'.format(bumped_version)
    call(['git', 'commit', '-m', msg])


if __name__ == '__main__':
    cli()
