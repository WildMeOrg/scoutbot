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
