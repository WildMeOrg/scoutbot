================
Wild Me ScoutBot
================

.. important::

   **This repository is superseded by** `scoutbot-two <https://github.com/SurveyTools/scoutbot-two>`_.

   Development, releases, and support continue in scoutbot-two, maintained by the Overhead Survey
   Workflow (OSW) team with Wild Me's support. The ``scoutbot`` package on PyPI is now published
   from scoutbot-two (version 0.5.0 and later); the 0.1.x releases remain installable but are no
   longer maintained. Docker images are no longer built from this repository.

   Documentation: https://surveytools.github.io/scoutbot-two/

|Docker| |ReadTheDocs| |Huggingface|

.. contents:: Quick Links
    :backlinks: none

.. sectnum::

How to Install
--------------

``pip install scoutbot`` now installs scoutbot-two (0.5.0 and later). To install the legacy
version documented here, pin below 0.5:

.. code-block:: bash

    pip install "scoutbot<0.5"

or, from source:

.. code-block:: bash

   git clone https://github.com/WildMeOrg/scoutbot
   cd scoutbot
   pip install -e .

To then add GPU acceleration, you need to replace `onnxruntime` with `onnxruntime-gpu`:

.. code-block:: bash

   pip uninstall -y onnxruntime
   pip install onnxruntime-gpu

How to Run
----------

You can run the tile-based Gradio demo with:

.. code-block:: bash

   python app.py

or, you can run the image-based Gradio demo with:

.. code-block:: bash

   python app2.py

To run with Docker:

.. code-block:: bash

    docker run \
       -it \
       --rm \
       -p 7860:7860 \
       -e CONFIG=phase1 \
       -e WIC_BATCH_SIZE=512 \
       --gpus all \
       --name scoutbot \
       wildme/scoutbot:main \
       python3 app2.py

To run with Docker Compose:

.. code-block:: yaml

    version: "3"

    services:
      scoutbot:
        image: wildme/scoutbot:main
        command: python3 app2.py
        ports:
          - "7860:7860"
        environment:
          CONFIG: phase1
          WIC_BATCH_SIZE: 512
        restart: unless-stopped
        deploy:
          resources:
            reservations:
              devices:
                - driver: nvidia
                  device_ids: ["all"]
                  capabilities: [gpu]

and run ``docker compose up -d``.

How to Build and Deploy
-----------------------

Docker Hub
==========

Previously built images remain on Docker Hub as ``wildme/scoutbot:latest``, but they are no longer
built or updated automatically from this repository. To build an image manually from this code,
use the code below:

.. code-block:: bash

    docker login

    export DOCKER_BUILDKIT=1
    export DOCKER_CLI_EXPERIMENTAL=enabled
    docker buildx create --name multi-arch-builder --use

    docker buildx build \
        -t wildme/scoutbot:latest \
        --platform linux/amd64 \
        --push \
        .

PyPI
====

This repository no longer publishes to PyPI. The `scoutbot <https://pypi.org/project/scoutbot/>`_
project is now owned by the OSW PyPI organization and released from
`scoutbot-two <https://github.com/SurveyTools/scoutbot-two>`_; see its
`release process <https://surveytools.github.io/scoutbot-two/development.html#releasing>`_.

Tests and Coverage
------------------

You can run the automated tests in the ``tests/`` folder by running:

.. code-block:: bash

    pip install -r requirements.optional.txt
    pytest

You may also get a coverage percentage by running:

.. code-block:: bash

    coverage html

and open the `coverage/html/index.html` file in your browser.

Building Documentation
----------------------

There is Sphinx documentation in the ``docs/`` folder, which can be built by running:

.. code-block:: bash

    cd docs/
    pip install -r requirements.optional.txt
    sphinx-build -M html . build/

Logging
-------

The script uses Python's built-in logging functionality called ``logging``.  All print functions are replaced with ``log.info()``, which sends the output to two places:

#. the terminal window, and
#. the file `scoutbot.log`

Code Formatting
---------------

It's recommended that you use ``pre-commit`` to ensure linting procedures are run on any code you write.  See `pre-commit.com <https://pre-commit.com/>`_ for more information.

Reference `pre-commit's installation instructions <https://pre-commit.com/#install>`_ for software installation on your OS/platform. After you have the software installed, run ``pre-commit install`` on the command line. Now every time you commit to this project's code base the linter procedures will automatically run over the changed files.  To run pre-commit on files preemtively from the command line use:

.. code-block:: bash

    pip install -r requirements.optional.txt
    pre-commit run --all-files

The code base has been formatted by `Brunette <https://pypi.org/project/brunette/>`_, which is a fork and more configurable version of `Black <https://black.readthedocs.io/en/stable/>`_.  Furthermore, try to conform to ``PEP8``.  You should set up your preferred editor to use ``flake8`` as its Python linter, but pre-commit will ensure compliance before a git commit is completed.  This will use the ``flake8`` configuration within ``setup.cfg``, which ignores several errors and stylistic considerations.  See the ``setup.cfg`` file for a full and accurate listing of stylistic codes to ignore.


.. |Docker| image:: https://img.shields.io/docker/image-size/wildme/scoutbot/latest
    :target: https://hub.docker.com/r/wildme/scoutbot
    :alt: Docker

.. |ReadTheDocs| image:: https://readthedocs.org/projects/scoutbot/badge/?version=latest
    :target: https://wildme-scoutbot.readthedocs.io/en/latest/?badge=latest
    :alt: ReadTheDocs

.. |Huggingface| image:: https://img.shields.io/badge/HuggingFace-running-success
    :target: https://huggingface.co/spaces/WildMeOrg/scoutbot
    :alt: Huggingface
