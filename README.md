---
title: Wild Me Scout
metaTitle: "The computer vision for Wild Me's Scout project"
emoji: 🌎
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 3.1.4
app_file: app.py
pinned: true
python_version: 3.10.5
---


Wild Me Scout
=============

[![GitHub CI](https://github.com/WildMeOrg/scoutbot/actions/workflows/testing.yml/badge.svg?branch=main)](https://github.com/WildMeOrg/scoutbot/actions/workflows/testing.yml)
[![Python Wheel](https://github.com/WildMeOrg/scoutbot/actions/workflows/python-publish.yml/badge.svg)](https://github.com/WildMeOrg/scoutbot/actions/workflows/python-publish.yml)
[![ReadTheDocs](https://readthedocs.org/projects/scoutbot/badge/?version=latest)](https://scoutbot.readthedocs.io/en/latest/?badge=latest)
[![Huggingface](https://img.shields.io/badge/HuggingFace-Running-yellow)](https://huggingface.co/spaces/WildMeOrg/scoutbot)

::: {.contents backlinks="none"}
Quick Links
:::

::: {.sectnum}
:::

How to Install
--------------

You need to first install Anaconda on your machine. Below are the
instructions on how to install Anaconda on an Apple macOS machine, but
it is possible to install on a Windows and Linux machine as well.
Consult the [official Anaconda page](https://www.anaconda.com) to
download and install on other systems. For Windows computers, it is
highly recommended that you intall the [Windows Subsystem for
Linux](https://docs.microsoft.com/en-us/windows/wsl/install).

``` {.bash}
# Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Anaconda and expose conda to the terminal
brew install anaconda
export PATH="/opt/homebrew/anaconda3/bin:$PATH"
conda init zsh
conda update conda
```

Once Anaconda is installed, you will need an environment and the
following packages installed

``` {.bash}
# Create Environment
conda create --name scout
conda activate scout

# Install Python dependencies
conda install pip

conda install -r requirements.txt
conda install pytorch torchvision -c pytorch-nightly
```

How to Run
----------

It is recommended to use [ipython]{.title-ref} and to copy sections of code
into and inspecting the

``` {.bash}
# Run the training script
cd scoutbot/
python train.py

# Run the live demo
python app.py
```

Unit Tests
----------

You can run the automated tests in the [tests/]{.title-ref} folder by
running [pytest]{.title-ref}. This will give an output of which tests
have failed. You may also get a coverage percentage by running [coverage
html]{.title-ref} and loading the [coverage/html/index.html]{.title-ref}
file in your browser. pytest

Building Documentation
----------------------

There is Sphinx documentation in the [docs/]{.title-ref} folder, which
can be built with the code below:

``` {.bash}
cd docs/
sphinx-build -M html . build/
```

Logging
-------

The script uses Python\'s built-in logging functionality called
[logging]{.title-ref}. All print functions are replaced with
[log.info]{.title-ref} within this script, which sends the output to two
places: 1) the terminal window, 2) the file [scout.log]{.title-ref}.
Get into the habit of writing text logs and keeping date-specific
versions for comparison and debugging.

Code Formatting
---------------

It\'s recommended that you use `pre-commit` to ensure linting procedures
are run on any code you write. (See also
[pre-commit.com](https://pre-commit.com/))

Reference [pre-commit\'s installation
instructions](https://pre-commit.com/#install) for software installation
on your OS/platform. After you have the software installed, run
`pre-commit install` on the command line. Now every time you commit to
this project\'s code base the linter procedures will automatically run
over the changed files. To run pre-commit on files preemtively from the
command line use:

``` {.bash}
git add .
pre-commit run

# or

pre-commit run --all-files
```

The code base has been formatted by Brunette, which is a fork and more
configurable version of Black
(<https://black.readthedocs.io/en/stable/>). Furthermore, try to conform
to PEP8. You should set up your preferred editor to use flake8 as its
Python linter, but pre-commit will ensure compliance before a git commit
is completed. This will use the flake8 configuration within `setup.cfg`,
which ignores several errors and stylistic considerations. See the
`setup.cfg` file for a full and accurate listing of stylistic codes to
ignore.

