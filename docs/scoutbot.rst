ScoutBot API
============

.. toctree::
   :maxdepth: 3
   :caption: Contents:

ScoutBot is the machine learning interface for the Wild Me Scout project.  This page specifies
the Python API to interact with all of the algorithms and machine learning models that have been
pretrained for inference in a production environment.

Overview
--------

In general, the structure of this API is to expose four main processing components for the Scout project.
These components are, in order: ``TILE``, ``WIC``, ``LOC``, and ``AGG``.

   1. ``TILE``: A module to convert images to tiles
   2. ``WIC``: A module to classify tiles as relevant for further processing (i.e., does it likely have an elephant?)
   3. ``LOC``: A module to detect elephants in tiles
   4. ``AGG``: A module to aggregate the tile-level detections back onto the original image

The ``TILE`` step and ``AGG`` steps are heuristic-based algorithms and do not need to use any
machine learning (ML) models or GPU offload.  In contrast, the ``WIC`` and ``LOC`` steps both require
their own ML models and can be computed on CPU or GPU (if available).

The non-ML components (``TILE`` and ``AGG``) both expose :func:`compute` functions, which is the single
point of interaction as the developer:

   - :meth:`scoutbot.tile.compute`
   - :meth:`scoutbot.agg.compute`

The ML components (``WIC`` and ``LOC``), in contrast, is a bit more complex and exposes three functions:

   - :func:`pre` (preprocessing)
   - :func:`predict` (inference)
   - :func:`post` (postprocessing)

For the WIC, these functions are:

   - :meth:`scoutbot.wic.pre`
   - :meth:`scoutbot.wic.predict`
   - :meth:`scoutbot.wic.post`

and for the LOC, these functions are:

   - :meth:`scoutbot.loc.pre`
   - :meth:`scoutbot.loc.predict`
   - :meth:`scoutbot.loc.post`

CDN Model Download (ONNX)
-------------------------

All of the machine learning models are hosted on GitHub as LFS files.  The two modules (``WIC`` and ``LOC``)
however need those files downloaded to the local machine prior to running inference.  These models are
hosted on a separate CDN for convenient access and can be fetched by running the following functions:

   - :meth:`scoutbot.wic.fetch`
   - :meth:`scoutbot.loc.fetch`

These functions will download the following files and will store them in your Operating System's default
cache folder:

   - ``WIC``: ``https://wildbookiarepository.azureedge.net/models/scout.wic.5fbfff26.3.0.onnx`` (81MB)
      SHA256 checksum: ``cbc7f381fa58504e03b6510245b6b2742d63049429337465d95663a6468df4c1``
   - ``LOC``: ``https://wildbookiarepository.azureedge.net/models/scout.loc.5fbfff26.0.onnx`` (209MB)
      SHA256 checksum: ``85a9378311d42b5143f74570136f32f50bf97c548135921b178b46ba7612b216``

Tiles (TILE)
------------

.. automodule:: scoutbot.tile
   :members:
   :undoc-members:
   :show-inheritance:


Whole-Image Classifier (WIC)
----------------------------

.. automodule:: scoutbot.wic
   :members:
   :undoc-members:
   :show-inheritance:

Localizer (LOC)
---------------

.. automodule:: scoutbot.loc
   :members:
   :undoc-members:
   :show-inheritance:

Aggregation (AGG)
-----------------

.. automodule:: scoutbot.agg
   :members:
   :undoc-members:
   :show-inheritance:

Pipeline (PIPE)
---------------

.. automodule:: scoutbot.__init__
   :members:
   :undoc-members:
   :show-inheritance:

Utilities
---------

.. automodule:: scoutbot.utils
   :members:
   :undoc-members:
   :show-inheritance:
