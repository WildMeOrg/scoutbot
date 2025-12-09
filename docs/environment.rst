Environment Variables
---------------------

The Scoutbot API and CLI have environment variables (envars) that allow you to configure global settings
and configurations.

   - ``CONFIG`` (default: mvp)
      The configuration setting for which machine lerning models to use.
      Must be one of ``phase1`` or ``mvp``, or their respective aliases as ``old`` or ``new``.
   - ``WIC_CONFIG`` (default: not set)
      The configuration setting for which machine lerning models to use with the WIC.
      Must be one of ``phase1`` or ``mvp``, or their respective aliases as ``old`` or ``new``.
      Defaults to the value of the ``CONFIG`` environment variable.
   - ``LOC_CONFIG`` (default: not set)
      The configuration setting for which machine lerning models to use with the LOC.
      Must be one of ``phase1`` or ``mvp``, or their respective aliases as ``old`` or ``new``.
      Defaults to the value of the ``CONFIG`` environment variable.
   - ``AGG_CONFIG`` (default: not set)
      The configuration setting for which machine lerning models to use with the AGG.
      Must be one of ``phase1`` or ``mvp``, or their respective aliases as ``old`` or ``new``.
      Defaults to the value of the ``CONFIG`` environment variable.
   - ``WIC_BATCH_SIZE`` (default: 160)
      The configuration setting for how many tiles to send to the GPU in a single batch during the WIC
      prediction (forward inference).  The LOC model has a fixed batch size (16 for ``phase1`` and
      32 for ``mvp``) and cannot be adjusted.  This setting can be used to control how fast the pipeline
      runs, as a trade-off of faster compute for more memory usage.  It is highly suggested to set this
      value as high as possible to fit into the GPU.
   - ``FAST`` (default: not set)
      A flag that can be set to turn off extracting the second grid of tiles.  Defaults to "not set", which
      translates to the standard process of extracting all tiles for grid1 and grid2.  Setting this
      value to anything will turn off grid2 and results in faster (but less accurate) detections
      (e.g., ``FAST=1``).
   - ``VERBOSE`` (default: not set)
      A verbosity flag that can be set to turn on debug logging.  Defaults to "not set", which translates
      to no debug logging.  Setting this value to anything will turn on debug logging
      (e.g., ``VERBOSE=1``).
   - ``SCOUTBOT_MODEL_URL`` (default: https://wildbookiarepository.azureedge.net/models)
      The base URL or path for accessing ONNX model files. This can be:

      * An HTTP/HTTPS URL (e.g., ``https://example.com/models``)
      * A local file path (e.g., ``/opt/models`` or ``C:\models``)
      * A network path (e.g., ``//server/share/models`` or ``\\server\share\models``)

      When using file paths, models will be copied to a local cache for consistency.
      This allows organizations to host models on their own infrastructure, CDN, or
      local/network storage.
   - ``SCOUTBOT_DATA_URL`` (default: https://wildbookiarepository.azureedge.net/data)
      The base URL or path for downloading test data files.

      As with SCOUTBOT_MODEL_URL, this can be:
      * An HTTP/HTTPS URL or
      * A local file path or
      * A network path

      This allows organizations to host test data on their own infrastructure, CDN, or
      local/network storage.
