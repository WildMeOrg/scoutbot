============
Known Issues
============

- Non-determinism and ONNX Runtime prediction failure on macOS when using MVP WIC
  model and a batch size greater than 1.  The code will automatically recude the
  batch size to 1 for this configuration and applicable environments.
