Testing
=======

Running tests
-------------

Each time you modify the codebase it's important that you make sure all
tests pass and that all files comply with the style guide:

::

    # this command will run tests and check the style in all files
    pytest --flake8

Modifying/adding your own tests
-------------------------------

If you are fixing a bug, chances are, you will need to update tests or
add more cases. Take a look at the `pytest
documentation <https://docs.pytest.org/en/latest/>`__
