# Deep Hough Transform for Semantic Line Detection

This repository contains the official PyTorch training and inference code, a Jittor inference port, data preparation scripts, evaluation scripts, and the LaTeX source for the paper _Deep Hough Transform for Semantic Line Detection_ (ECCV 2020, TPAMI 2021).

Project links:

- Paper: [arXiv 2003.04676](https://arxiv.org/abs/2003.04676)
- Online demo: [mc.nankai.edu.cn/dht](http://mc.nankai.edu.cn/dht)
- Project page: [mmcheng.net/dhtline](http://mmcheng.net/dhtline)
- NKL dataset: [download page](https://data.kaizhao.net/deep-hough-transform/NKL.zip)
- Line annotator: [Hanqer/lines-manual-labeling](https://github.com/Hanqer/lines-manual-labeling)

The repository is organized around one core idea: an image backbone extracts multi-scale features, Deep Hough Transform layers project those features into Hough space, and the final heatmap is decoded into semantic line segments. The PyTorch path is used for training, validation, and metric evaluation. The Jittor path mirrors the inference pipeline and is used for speed testing.

## Overview

The typical workflow is:

1. Prepare the dataset split files and convert line annotations into Hough-space labels.
2. Configure the experiment in `config.yml`.
3. Train with `train.py`.
4. Run inference with `forward.py` or the Jittor equivalent in `jittor_code/forward.py`.
5. Evaluate predictions with `test_nkl.py` or `test_sel.py`.

The repository also includes the paper source under `LaTeX/`, reusable geometric and evaluation utilities, and legacy compiled-extension sources under `model/_cdht/` and `chamfer_distance/`.

## Repository Layout

### Top-level files

- `README.md`: Main project guide and repository map.
- `train.py`: PyTorch training entry point. It loads the config, builds the model, creates the training and validation loaders, runs optimization, writes TensorBoard scalars, and saves checkpoints.
- `forward.py`: PyTorch inference entry point. It loads a checkpoint, runs test-time prediction, converts Hough-space responses back to line segments, draws visualizations, and saves `.npy` prediction files.
- `test_nkl.py`: Evaluation script for the NKL dataset. It reads saved `.npy` predictions, compares them with ground-truth text files, and exports precision, recall, and F-score curves.
- `test_sel.py`: Evaluation script for the SEL / ICCV2017 JTLEE dataset. It performs the same matching-based evaluation as `test_nkl.py`, but with the SEL annotation format.
- `dataloader.py`: PyTorch dataset and dataloader helpers for training, validation, and test-time inference.
- `basic_ops.py`: Geometry and annotation primitives for semantic lines, including line representation, Hough-label generation, masks, and boundary intersection helpers.
- `utils.py`: Conversion and post-processing utilities for line detection, including Hough mapping, reverse mapping, visualization, and edge alignment.
- `metric.py`: Line-comparison metrics such as endpoint alignment, Chamfer distance, and EMD-based scoring.
- `hungarian_matching.py`: Bipartite matching logic used to convert line-level similarity into TP/FP/FN counts.
- `logger.py`: Minimal logger that mirrors messages to stdout and a log file.
- `config.yml`: Default experiment configuration for the PyTorch pipeline.
- `requirements.txt`: Python dependencies needed for the PyTorch code path and the evaluation scripts.
- `test.md`: Short reproduction note for evaluating the NKL checkpoint.
- `pipeline.png`: Architecture diagram referenced by the original paper and README.
- `precision.csv`, `recall.csv`, `fscore.csv`: Metric exports produced by `test_nkl.py` and `test_sel.py`.
- `model_best_sel.pth`: Pretrained PyTorch checkpoint for the SEL setting.
- `dht_r50_nkl_d97b97138.pth`: Pretrained PyTorch checkpoint used for the NKL demo and inference.
- `results/`: Saved logs and experiment artifacts from previous runs.
- `result/`: Default experiment output root used by the config and training scripts.
- `.gitignore`: Excludes local environments, caches, archives, checkpoints, and generated outputs from version control.

### `data/`

This folder contains the data-preparation scripts and split lists. The dataset contents themselves are intentionally excluded from this guide.

- `prepare_data_JTLEE.py`: Converts the ICCV2017 JTLEE / SEL annotations into Hough-space training labels and saves per-image `.npy` metadata.
- `prepare_data_NKL.py`: Builds Hough-space labels for the NKL dataset.
- `Readme.txt`: Dataset-specific notes kept with the original data release.
- `train.txt`: Training split list for the SEL pipeline.
- `val.txt`: Validation split list.
- `train_idx_1716.txt`: Index list used for the 1716-image SEL split.
- `test_idx_1716.txt`: Test index list used for the 1716-image SEL split.
- `sel_test.txt`: Alternative SEL test split list.
- `training/`: Output directory for generated training labels and split metadata.

### `model/`

This directory contains the PyTorch model definition and the optional low-level DHT extension sources.

- `network.py`: Defines the main `Net` class. It selects a backbone, applies several DHT layers at different feature scales, upsamples all Hough feature maps to a shared resolution, concatenates them, and predicts the final line heatmap with a `1x1` convolution.
- `dht.py`: Wraps the Deep Hough Transform block as a reusable PyTorch module. It combines a channel-reduction convolution, the DHT operator, and post-DHT convolutions.
- `backbone/fpn.py`: ResNet-style Feature Pyramid Network implementation. It provides the multi-scale backbone used by the main model.
- `backbone/resnet.py`: ResNet and ResNet-101 backbone implementation with feature-pyramid outputs.
- `backbone/mobilenet.py`: MobileNetV2-based FPN backbone for a lighter model variant.
- `backbone/vgg_fpn.py`: VGG16-based FPN backbone.
- `backbone/res2net.py`: Res2Net backbone with FPN outputs.
- `_cdht/setup.py`: Build script for the original CUDA extension named `deep_hough`.
- `_cdht/dht_func.py`: Python wrapper around the DHT operator. In this repository state it falls back to the pure PyTorch CPU implementation rather than the compiled CUDA module.
- `_cdht/dht_cpu.py`: Pure PyTorch implementation of the DHT accumulation operator used as the active fallback path.
- `_cdht/deep_hough_cuda.cpp`: C++ binding for the original CUDA extension.
- `_cdht/deep_hough_cuda_kernel.cu`: CUDA kernels that implement the forward and backward DHT accumulation logic.
- `_cdht/.gitignore`: Ignores build artifacts created while compiling the extension.

### `chamfer_distance/`

This folder contains a lightweight Chamfer-distance implementation and the legacy C++ / CUDA sources that were used in earlier versions of the project.

- `__init__.py`: Exports `ChamferDistance` for convenience.
- `chamfer_distance.py`: Current pure-PyTorch Chamfer-distance module used by `metric.py`.
- `chamfer_distance.cpp`: Legacy C++ extension source.
- `chamfer_distance.cu`: Legacy CUDA extension source.

### `jittor_code/`

This directory mirrors the PyTorch pipeline in Jittor. It is focused on inference and benchmarking rather than training.

- `README.md`: Jittor-specific usage notes, speed table, and reproduction steps.
- `config.yml`: Jittor configuration file.
- `forward.py`: Jittor inference script. It loads a checkpoint, runs test-time prediction, supports optional hook-based dumping, and reports throughput.
- `benchmark.py`: Small benchmark script that measures raw Jittor inference throughput.
- `basic_ops.py`: Jittor-flavored port of the line geometry helpers from the top-level `basic_ops.py`.
- `dataloader.py`: Jittor dataset wrapper for test-time loading.
- `logger.py`: Jittor-side logger utility, matching the top-level logger behavior.
- `utils.py`: Jittor-flavored utility helpers for line mapping and visualization.
- `model/network.py`: Jittor version of the main `Net` architecture.
- `model/dht.py`: Jittor Deep Hough Transform module and autograd wrapper.
- `model/cuda_src.py`: CUDA source templates embedded as Python strings for the Jittor DHT implementation.
- `model/backbone/fpn.py`: Jittor FPN backbone used by the inference model.

### `LaTeX/`

This folder contains the paper source and build helpers.

- `line-tpami.tex`: Main LaTeX manuscript for the TPAMI version of the paper.
- `line.bib`: Bibliography file.
- `build.sh`: Builds the PDF from the LaTeX source.
- `clean.sh`: Removes auxiliary LaTeX build artifacts.
- `readme.md`: Notes describing the LaTeX source tree.
- `figures/`: Figures referenced by the paper source.

## How The Main Scripts Fit Together

### Training pipeline

`train.py` loads `config.yml`, creates the model from `model/network.py`, and consumes training data through `dataloader.py`. The network predicts a Hough-space heatmap, and the script optimizes it with binary cross-entropy against the precomputed labels saved by the data-preparation scripts. During training, it records scalar summaries, logs progress, and writes checkpoints into the configured output directory.

### PyTorch inference pipeline

`forward.py` restores a checkpoint, switches the model to evaluation mode, and runs prediction on the configured test set. The output heatmap is thresholded, connected components are extracted, Hough-space centroids are mapped back into image-space line segments with `utils.reverse_mapping`, and the resulting lines are drawn and saved. The script also writes `.npy` arrays containing the predicted coordinates.

### Evaluation pipeline

`test_nkl.py` and `test_sel.py` compare saved predictions against ground-truth line annotations. Both scripts iterate over thresholds from `0.01` to `0.99`, call `hungarian_matching.caculate_tp_fp_fn`, and derive precision, recall, and F-score curves. The NKL script saves CSV files with the curves; the SEL script prints the same summary metrics and leaves the CSV export commented out.

### Jittor inference pipeline

The Jittor code under `jittor_code/` mirrors the PyTorch inference flow and is intended for speed comparison. The `forward.py` script can also attach hooks for dumping intermediate tensors, and `benchmark.py` measures raw throughput without the post-processing stage.

## Core Concepts Used Across The Codebase

- A line is stored as `[y0, x0, y1, x1]`.
- Hough-space labels are generated on a fixed angular and radial grid, controlled by `NUMANGLE` and `NUMRHO` in the config.
- The model predicts a single-channel heatmap in Hough space, where connected components correspond to candidate line hypotheses.
- Candidate lines are mapped back to image space, clipped to image boundaries, and optionally refined with edge alignment.
- Final evaluation uses Hungarian matching so the predicted set and ground-truth set are compared as sets of line instances rather than as independent pixels.

## Configuration

The default PyTorch configuration in `config.yml` defines:

- dataset directories and split files,
- batch size and worker count,
- optimizer hyperparameters,
- the Hough-grid resolution,
- the backbone choice,
- the prediction threshold,
- and the output directory used for logs and checkpoints.

The supported backbone names in the main model are:

`resnet18`, `resnet50`, `resnet101`, `resnext50`, `res2net50`, `mobilenetv2`, and `vgg16`.

## Build And Run

### Install dependencies

```sh
pip install -r requirements.txt
```

### Prepare labels

For SEL / JTLEE:

```sh
python data/prepare_data_JTLEE.py
  --root './data/ICCV2017_JTLEE_images/'
  --label './data/ICCV2017_JTLEE_gtlines_all'
  --save-dir './data/training/JTLEE_resize_100_100/'
  --list './data/training/JTLEE.lst'
  --prefix 'JTLEE_resize_100_100'
  --fixsize 400 --numangle 100 --numrho 100
```

For NKL:

```sh
python data/prepare_data_NKL.py
  --root './data/NKL'
  --label './data/NKL'
  --save-dir './data/training/NKL_resize_100_100'
  --fixsize 400
```

### Train

```sh
python train.py
```

### Run inference

```sh
python forward.py --model model_best_sel.pth --tmp ./result/reproduce
```

### Evaluate predictions

```sh
python test_nkl.py --pred ./result/reproduce/visualize_test --gt ./data/NKL
python test_sel.py --pred ./result/reproduce/visualize_test --gt ./data/training/JTLEE_resize_100_100
```

### Jittor inference

```sh
cd jittor_code
python forward.py --model ../model_best_sel.pth --tmp ../result/jittor
```

## Notes On The Low-Level DHT Code

The original CUDA extension sources are still included in `model/_cdht/`, but the active PyTorch wrapper in this repository currently uses the pure PyTorch fallback in `dht_cpu.py`. That keeps the code runnable even when the compiled extension is not built. The CUDA sources are still useful as the reference implementation and for anyone who wants to rebuild the original extension.

## Citation

If this project is useful in your research, please cite:

```bibtex
@article{zhao2021deep,
  author    = {Kai Zhao and Qi Han and Chang-bin Zhang and Jun Xu and Ming-ming Cheng},
  title     = {Deep Hough Transform for Semantic Line Detection},
  journal   = {IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
  year      = {2021},
  doi       = {10.1109/TPAMI.2021.3077129}
}

@inproceedings{eccv2020line,
  title={Deep Hough Transform for Semantic Line Detection},
  author={Qi Han and Kai Zhao and Jun Xu and Ming-Ming Cheng},
  booktitle={ECCV},
  pages={750--766},
  year={2020}
}
```

## License

This project is released under the Creative Commons NonCommercial license, CC BY-NC 3.0. Commercial use requires direct permission from the authors.
