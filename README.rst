================
Wild Me ScoutBot
================

|Tests| |Codecov| |Wheel| |Docker| |ReadTheDocs| |Huggingface|

.. contents:: Quick Links
    :backlinks: none

.. sectnum::

How to Install
--------------

You need to first install Anaconda on your machine.  Below are the instructions on how to install Anaconda on an Apple macOS machine, but it is possible to install on a Windows and Linux machine as well.  Consult the `official Anaconda page <https://www.anaconda.com>`_ to download and install on other systems.

.. code:: bash

   # Install Homebrew
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

   # Install Anaconda and expose conda to the terminal
   brew install anaconda
   export PATH="/opt/homebrew/anaconda3/bin:$PATH"
   conda init zsh
   conda update conda

Once Anaconda is installed, you will need an environment and the following packages installed

.. code:: bash

   # Create Environment
   conda create --name scoutbot
   conda activate scoutbot

   # Install Python dependencies
   conda install pip

   pip install -r requirements.txt
   # conda install pytorch torchvision -c pytorch-nightly

How to Run
----------

It is recommended to use `ipython` and to copy sections of code into and inspecting the

.. code:: bash

   # Run the live demo
   python app.py

Docker
------

The application can also be built into a Docker image and hosted on Docker Hub.

.. code:: bash

    # linux/amd64

    docker login

    export DOCKER_BUILDKIT=1
    export DOCKER_CLI_EXPERIMENTAL=enabled
    docker buildx create --name multi-arch-builder --use

    docker buildx build \
        -t wildme/scoutbot:latest \
        --platform linux/arm64 \
        --push \
        .

To run:

.. code:: bash

    docker run \
       -it \
       --rm \
       -p 7860:7860 \
       --name scoutbot \
       wildme/scoutbot:latest

Unit Tests
----------

You can run the automated tests in the `tests/` folder by running `pytest`.  This will give an output of which tests have failed.  You may also get a coverage percentage by running `coverage html` and loading the `coverage/html/index.html` file in your browser.
pytest

Building Documentation
----------------------

There is Sphinx documentation in the `docs/` folder, which can be built with the code below:

.. code:: bash

    cd docs/
    sphinx-build -M html . build/

Logging
-------

The script uses Python's built-in logging functionality called `logging`.  All print functions are replaced with `log.info` within this script, which sends the output to two places: 1) the terminal window, 2) the file `scoutbot.log`.  Get into the habit of writing text logs and keeping date-specific versions for comparison and debugging.

Code Formatting
---------------

It's recommended that you use ``pre-commit`` to ensure linting procedures are run
on any code you write. (See also `pre-commit.com <https://pre-commit.com/>`_)

Reference `pre-commit's installation instructions <https://pre-commit.com/#install>`_ for software installation on your OS/platform. After you have the software installed, run ``pre-commit install`` on the command line. Now every time you commit to this project's code base the linter procedures will automatically run over the changed files.  To run pre-commit on files preemtively from the command line use:

.. code:: bash

    git add .
    pre-commit run

    # or

    pre-commit run --all-files

The code base has been formatted by Brunette, which is a fork and more configurable version of Black (https://black.readthedocs.io/en/stable/).  Furthermore, try to conform to PEP8.  You should set up your preferred editor to use flake8 as its Python linter, but pre-commit will ensure compliance before a git commit is completed.  This will use the flake8 configuration within ``setup.cfg``, which ignores several errors and stylistic considerations.  See the ``setup.cfg`` file for a full and accurate listing of stylistic codes to ignore.


.. |Tests| image:: https://github.com/WildMeOrg/scoutbot/actions/workflows/testing.yml/badge.svg?branch=main
    :target: https://github.com/WildMeOrg/scoutbot/actions/workflows/testing.yml
    :alt: GitHub CI

.. |Codecov| image:: https://codecov.io/gh/WildMeOrg/scoutbot/branch/main/graph/badge.svg?token=FR6ITMWQNI
    :target: https://codecov.io/gh/WildMeOrg/scoutbot
    :alt: Codecov

.. |Wheel| image:: https://github.com/WildMeOrg/scoutbot/actions/workflows/python-publish.yml/badge.svg
    :target: https://github.com/WildMeOrg/scoutbot/actions/workflows/python-publish.yml
    :alt: Python Wheel

.. |Docker| image:: https://img.shields.io/docker/image-size/wildme/scoutbot/latest
    :target: https://hub.docker.com/r/wildme/scoutbot
    :alt: Docker

.. |ReadTheDocs| image:: https://readthedocs.org/projects/scoutbot/badge/?version=latest
    :target: https://wildme-scoutbot.readthedocs.io/en/latest/?badge=latest
    :alt: ReadTheDocs

.. |Huggingface| image:: https://img.shields.io/badge/HuggingFace-running-success
    :target: https://huggingface.co/spaces/WildMeOrg/scoutbot
    :alt: Huggingface
