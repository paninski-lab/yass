Developer's Guide
=================

Internal contributors
---------------------

Internal contributors have write access to the repo, we are following this model:

```
# the first thing to do is to clone the repo
git clone https://github.com/paninski-lab/yass

# install the package in editable mode so your changes take effect when
# you import the package from the Python interpreter
cd yass
pip install --editable .

# move to the dev branch
git checkout dev

# when you start working on something new, create a new branch from dev
git checkout -b new-feature

# work on new feature...

# remember to push you changes to the remote branch
git push

# when the new feature is done open a pull request to merge new-feature to dev

# once the pull request is accepted and merged to dev, don't forget to remove
# the branch if you no longer are going to use it
# remove from the remote repository
git push -d origin new-feature
# remove from your local repository
git brancg -d new-feature
```

