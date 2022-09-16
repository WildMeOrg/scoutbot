================
Wild Me ScoutBot
================

|Tests| |Codecov| |Wheel| |Docker| |ReadTheDocs| |Huggingface|

.. contents:: Quick Links
    :backlinks: none

.. sectnum::

How to Install
--------------

.. code-block:: console

    (.venv) $ pip install scoutbot

or, from source:

.. code-block:: console

   git clone https://github.com/WildMeOrg/scoutbot
   cd scoutbot
   (.venv) $ pip install -e .

How to Run
----------

You can run the tile-base Gradio demo with:

.. code-block:: console

   (.venv) $ python app.py

or, you can run the image-base Gradio demo with:

.. code-block:: console

   (.venv) $ python app2.py

Docker
------

The application can also be built into a Docker image and is hosted on Docker Hub as ``wildme/scoutbot:latest``.

.. code-block:: console

    docker login

    export DOCKER_BUILDKIT=1
    export DOCKER_CLI_EXPERIMENTAL=enabled
    docker buildx create --name multi-arch-builder --use

    docker buildx build \
        -t wildme/scoutbot:latest \
        --platform linux/amd64 \
        --push \
        .

To run with Docker:

.. code-block:: console

    docker run \
       -it \
       --rm \
       -p 7860:7860 \
       --name scoutbot \
       wildme/scoutbot:latest

Tests and Coverage
------------------

You can run the automated tests in the ``tests/`` folder by running:

.. code-block:: console

    (.venv) $ pip install -r requirements.optional.txt
    (.venv) $ pytest

You may also get a coverage percentage by running:

.. code-block:: console

    (.venv) $ coverage html

and open the `coverage/html/index.html` file in your browser.

Building Documentation
----------------------

There is Sphinx documentation in the ``docs/`` folder, which can be built by running:

.. code-block:: console

    (.venv) $ cd docs/
    (.venv) $ pip install -r requirements.optional.txt
    (.venv) $ sphinx-build -M html . build/

Logging
-------

The script uses Python's built-in logging functionality called ``logging``.  All print functions are replaced with :func:``log.info``, which sends the output to two places:

    - 1. the terminal window, and
    - 2. the file `scoutbot.log`

Code Formatting
---------------

It's recommended that you use ``pre-commit`` to ensure linting procedures are run
on any code you write.  See `pre-commit.com <https://pre-commit.com/>`_ for more information.

Reference `pre-commit's installation instructions <https://pre-commit.com/#install>`_ for software installation on your OS/platform. After you have the software installed, run ``pre-commit install`` on the command line. Now every time you commit to this project's code base the linter procedures will automatically run over the changed files.  To run pre-commit on files preemtively from the command line use:

.. code-block:: console

    (.venv) $ pip install -r requirements.optional.txt
    (.venv) $ pre-commit run --all-files

The code base has been formatted by `Brunette <https://pypi.org/project/brunette/>`_, which is a fork and more configurable version of `Black <https://black.readthedocs.io/en/stable/>`_.  Furthermore, try to conform to ``PEP8``.  You should set up your preferred editor to use ``flake8`` as its Python linter, but pre-commit will ensure compliance before a git commit is completed.  This will use the ``flake8`` configuration within ``setup.cfg``, which ignores several errors and stylistic considerations.  See the ``setup.cfg`` file for a full and accurate listing of stylistic codes to ignore.


.. |Tests| image:: https://github.com/WildMeOrg/scoutbot/actions/workflows/testing.yml/badge.svg?branch=main
    :target: https://github.com/WildMeOrg/scoutbot/actions/workflows/testing.yml
    :alt: GitHub CI

.. |Codecov| image:: https://codecov.io/gh/WildMeOrg/scoutbot/branch/main/graph/badge.svg?token=FR6ITMWQNI
    :target: https://app.codecov.io/gh/WildMeOrg/scoutbot
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
