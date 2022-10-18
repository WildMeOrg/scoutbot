=========
Changelog
=========

Version 0.1.17
--------------

* **[BREAKING CHANGE]** Added detection label mapping for the ``phase1`` output to rename the following species labels to be consistent with the ``mvp`` output labels:

    * ``elephant_savanna`` to ``elephant``
    * ``dead_animalwhite_bones`` to ``white_bones``
    * ``deadbones`` to ``white_bones``
    * ``elecarcass_old`` to ``white_bones``
    * ``gazelle_gr`` to ``gazelle_grants``
    * ``gazelle_th`` to ``gazelle_thomsons``

* Added rounding to the WIC predicted confidence to 4 decimal points in the print and JSON outputs.
* Added to the documentation the list of supported class labels for each model configuration.
* Updated the documentation to highlight the list of recommended supported species for the MVP model
  (based on held-out validation data), and to add the build and deployment instructions.
* Added platform detection code to detect macOS and reduce the batch size of WIC models with the
  MVP model to 1 (added to Known Issues).
* Added three new environment variables to allow specifying the model configuration for the ``WIC``,
  ``LOC``, and ``AGG``, respectively: ``WIC_CONFIG``, ``LOC_CONFIG``, ``AGG_CONFIG``.  If unset, it
  uses the global config and behavior as specified by the ``CONFIG`` environment variable.  The TILE
  module does not have different settings dependent on the model configuration.
* Added a new environment variable to allow for faster but less accurate results: ``FAST``.  If unset, it
  uses the standard tile extraction behavior for grid1 and grid2.  Turning this flag on will dramatically
  speed up inference by processing approximately half of the number of tiles per image.
* Added ``CHANGELOG.rst`` and ``ISSUES.rst``.
* Modified documentation strings in a few places for clarity and correctness.
* Updated both Gradio apps to support the Phase1 and MVP models and their respective configurations.
* Updated tests to support the new ML configuration default settings.

Version 0.1.16
--------------

*Alpha version of Scoutbot, with all Phase 1 and MVP functionality and pre-trained models included*
