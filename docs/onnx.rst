.. include:: colors.rst

CDN Model Download (ONNX)
-------------------------

All of the machine learning models are hosted on GitHub as LFS files.  The two modules (``WIC`` and ``LOC``)
however need those files downloaded to the local machine prior to running inference.  These models are
hosted on a separate CDN for convenient access and can be fetched by running the following functions:

   - :meth:`scoutbot.wic.fetch`
   - :meth:`scoutbot.loc.fetch`

To pre-download the models for a specific config (e.g., ``mvp``), you can specify an optional config:

   - :obj:`scoutbot.wic.fetch(config="mvp")`
   - :obj:`scoutbot.loc.fetch(config="mvp")`

These functions will download the following files and will store them in your Operating System's default
cache folder:

   - Phase 1: ``phase1``
      - WIC: ``https://wildbookiarepository.azureedge.net/models/scout.wic.5fbfff26.3.0.onnx`` (81MB)
         SHA256 checksum: ``cbc7f381fa58504e03b6510245b6b2742d63049429337465d95663a6468df4c1``
      - LOC: ``https://wildbookiarepository.azureedge.net/models/scout.loc.5fbfff26.0.onnx`` (194M)
         SHA256 checksum: ``85a9378311d42b5143f74570136f32f50bf97c548135921b178b46ba7612b216``

   - MVP: ``mvp``
      - WIC: ``https://wildbookiarepository.azureedge.net/models/scout.wic.mvp.2.0.onnx`` (97MB)
         SHA256 checksum: ``3ff3a192803e53758af5e112526ba9622f1dedc55e2fa88850db6f32af160f32``
      - LOC: ``https://wildbookiarepository.azureedge.net/models/scout.loc.mvp.0.onnx`` (194M)
         SHA256 checksum: ``f5bd22fbacc91ba4cf5abaef5197d1645ae5bc4e63e88839e6848c48b3710c58``

Supported Objects of Interest
-----------------------------

The ONNX models are pre-configured to support a specific batch size and will predict specific species in
the final detection results.  The input sizes are defined explicitly when they cannot be changed, but the
``WIC`` model's inputs can be balanced using the environment variable ``WIC_BATCH_SIZE``.  The outputs of
the pipeline is a collection of bounding boxes, confidence values, and class labels.  Some of the labels
are not clean and are mapped, for convience, when the final detection labels are created.  Below are the
supported species for each model:

   - Phase 1: ``phase1``
      - :green:`elephant_savanna`
      - - mapped to: `elephant`

   - MVP: ``mvp``
      - :green:`buffalo`
      - `camel`
      - `canoe`
      - `car`
      - :green:`cow`
      - `crocodile`
      - `dead_animalwhite_bones`
      - - mapped to: `white_bones`
      - `deadbones`
      - - mapped to: `white_bones`
      - `eland`
      - `elecarcass_old`
      - - mapped to: `white_bones`
      - :green:`elephant`
      - `gazelle_gr`
      - - mapped to: `gazelle_grants`
      - `gazelle_grants`
      - `gazelle_th`
      - - mapped to: `gazelle_thomsons`
      - `gazelle_thomsons`
      - `gerenuk`
      - `giant_forest_hog`
      - :green:`giraffe`
      - `goat`
      - :green:`hartebeest`
      - :green:`hippo`
      - `impala`
      - :green:`kob`
      - `kudu`
      - `motorcycle`
      - `oribi`
      - :green:`oryx`
      - `ostrich`
      - `roof_grass`
      - `roof_mabati`
      - `sheep`
      - `test`
      - :green:`topi`
      - `vehicle`
      - :green:`warthog`
      - :green:`waterbuck`
      - `white_bones`
      - `wildebeest`
      - :green:`zebra`

All species above that are highlighted in green have an Average Precision (AP) of at least 50%.  The other species are supported in a preliminary sense and should not be heavily relied on.
