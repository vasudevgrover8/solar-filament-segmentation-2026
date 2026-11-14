# Inference artifacts

Download the `model-artifacts.zip` asset from this repository's **Releases → v1.0.0** and extract its contents into this directory. The archive must contain the exact exported `maskrcnn_weights.pt`, `tiled_cnn_weights.pt`, every crop CNN fold named in `inference_config.json`, `lightgbm_model.txt`, `lightgbm_without_crop.txt`, `inference_config.json`, and `runtime.txt`.

The reproduction notebook runs from these files and the competition test images. The full training checkpoint is only needed to rerun Part 2 training.
