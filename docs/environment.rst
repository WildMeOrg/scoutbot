Environment Variables
---------------------

The Scoutbot API and CLI have two environment variables (envars) that allow you to configure global settings
and configurations.

   - ``CONFIG`` (default: phase1)
      The configuration setting for which machine lerning models to use.
      Must be one of ``phase1`` or ``mvp``.
   - ``WIC_BATCH_SIZE`` (default: 256)
      The configuration setting for how many tiles to send to the GPU in a single batch during the WIC
      prediction (forward inference).  The LOC model has a fixed batch size (16 for ``phase1`` and
      32 for ``mvp``) and cannot be adjusted.  This setting can be used to control how fast the pipeline
      runs, as a trade-off of faster compute for more memory usage.  It is highly suggested to set this
      value as high as possible to fit into the GPU.
