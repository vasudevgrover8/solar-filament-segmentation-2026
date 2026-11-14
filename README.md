# Solar filament instance segmentation

Vasudev Grover · Solar Filament Segmentation Challenge 2026

The scored entry combines a frozen Mask R-CNN with a tiled CNN at 384 × 384, a candidate-crop CNN and an IoU-weighted LightGBM verifier. It assigns shared pixels to the higher-ranked mask before exporting the final CSV. Kaggle displayed a **public PQ of 0.33**.

| Result | PQ |
| --- | ---: |
| Labeled DEV holdout, IoU > 0.50 | 0.39743 |
| Labeled DEV holdout, IoU > 0.40 | 0.43124 |
| Kaggle public leaderboard, corrected CSV | 0.33 |

The DEV scores describe pre-correction predictions on 51 held-out images and 86 annotation records. The public result comes from Kaggle's scoring of the corrected test predictions.

## Reproduce the scored predictions

1. Download and extract the [competition data](https://www.kaggle.com/competitions/filament-segmentation-2026/data) under `data/`. The inference notebook looks for the `test_images` directory below it.
2. Download the [release asset](../../releases/tag/v1.0.0) and extract it into `models/`.
3. Use Python 3.10 and a CUDA-capable NVIDIA GPU. Inspect `models/runtime.txt` for the original environment. Install the compatible CUDA build of PyTorch and torchvision first, then `python -m pip install -r requirements.txt`. For the documented local CUDA 11.8 pair, use:

   ```sh
   python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
   ```

4. Open [reproduce_submission.ipynb](notebooks/reproduce_submission.ipynb) from the repository root or `notebooks/` and run all cells. It loads the exported weights, runs test inference, resolves shared pixels, and compares `outputs/reproduced/submission_no_overlap.csv` byte for byte with [the scored CSV](results/submission.csv). Exact replay was not independently run on a second GPU environment.

The CSV postprocessing can also be reproduced without a GPU:

```sh
python remove_overlap.py results/predictions_raw.csv outputs_submission.csv
```

The corrected reference retains 1,177 masks across 178 images with detections. Nine masks were trimmed across eight images. The per-image foreground union and all instances were retained.

## Repository guide

- [Part 1 exploration](notebooks/part1_exploration.ipynb) contains the saved exploratory analyses and outputs. Rerunning its analyses uses the original competition and feature-cache inputs described there.
- [Part 2 training and evaluation](notebooks/part2_training_fusion.ipynb) uses the original Mask R-CNN training checkpoint, fits the tiled and crop CNNs and LightGBM, and reports DEV ablations.
- [Technical report](report/technical_report.pdf) documents the pipeline, metrics, case analysis and submission format.
- [Submission manifest](results/submission_record.json) records the displayed leaderboard score and CSV checksum.
