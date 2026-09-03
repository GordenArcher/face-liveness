# ML Pipeline

This directory owns the face liveness / presentation attack detection
work. The rest of the project should treat this layer as the place where
models are trained, evaluated, thresholded, and eventually exported.

The current production boundary is not Go or gRPC yet. The current
boundary is a trained liveness model that can answer one question for an
already-cropped face image:

```
should recognition continue, or should this image be rejected as spoof?
```

## Setup

Use Python 3.12 and the local venv:

```
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## NUAA

NUAA is the first dataset used by this project. It is small and mostly
print-attack focused, so it is useful for building the pipeline but not
enough to justify service work by itself.

Expected local layout:

```
ml/data/raw/
  ClientFace/
  ImposterFace/
```

Train and evaluate the classical baseline:

```
python src/train_baseline.py
python src/evaluate.py
```

Train and evaluate the CNN on the same saved split:

```
python src/train_cnn.py --epochs 15
python src/evaluate_cnn.py
```

Run subject-disjoint k-fold validation:

```
python src/cross_validate_baseline.py --n-splits 5
python src/cross_validate_cnn.py --n-splits 5 --epochs 15
```

Current NUAA 5-fold result:

| Model         | ACER              | APCER             | BPCER             |
| ------------- | ----------------- | ----------------- | ----------------- |
| LBP + SVM     | 0.1878 +/- 0.0280 | 0.1031 +/- 0.0919 | 0.2725 +/- 0.0851 |
| CNN from zero | 0.0637 +/- 0.0316 | 0.1013 +/- 0.0720 | 0.0261 +/- 0.0293 |

The CNN is the better NUAA model, but APCER is still around 10% on
average. For a security gate, that is the part to treat with caution.

## Thresholds

The default CNN threshold is `0.5`. A stricter threshold lowers spoof
acceptance but raises live-user rejection:

```
python src/evaluate_cnn.py --threshold 0.7
python src/sweep_cnn_threshold.py
python src/sweep_cnn_threshold.py --max-apcer 0.05
```

Thresholds are selected from validation metrics first and then reported
on test. Do not choose a policy directly from test results; that would
overfit the test set.

## Local Inference

Use this to test one already-cropped face image against the saved model
artifacts:

```
python src/predict_liveness.py path/to/face.jpg --model cnn
python src/predict_liveness.py path/to/face.jpg --model baseline
```

This is a local ML boundary, not the final service interface. Full-frame
face detection, alignment, embeddings, gRPC, and Go integration come
after model validation.

## CelebA-Spoof

CelebA-Spoof is the next generalization check because it is larger and
covers richer spoof media than NUAA. The official dataset has 625,537
images from 10,177 subjects and 43 annotations. Its JSON annotation
index `40` is spoof type: `0` means live, and non-zero values are spoof
media. This project converts that to live = `1`, spoof = `0`.

The dataset is distributed for non-commercial research/education use.
The download helper therefore requires an explicit acknowledgement flag.

Download and prepare it:

```
./download_celeba_spoof.sh --i-accept-non-commercial-research-terms
```

Equivalent Python command:

```
python src/download_celeba_spoof.py --i-accept-non-commercial-research-terms
```

The downloader uses `gdown` against the official Google Drive folder,
passes `--remaining-ok` because the official folder is published as many
split files, assembles `CelebA_Spoof.zip.001` style parts into a normal
zip, extracts it under:

```
ml/data/celeba_spoof/
```

and validates that train, validation, and test annotation JSON files are
discoverable.

Assembly and extraction print a start line before the long disk work
begins, then percentage progress while the job runs. If the process is
still healthy, expect periodic lines such as `archive: assembling ...`,
`assembly: 12.34%`, or `extraction 1200/625000 files: 45.67%`. Long
silence after those phases start usually means the external volume is
asleep, disconnected, or blocked on disk I/O.

Downloader logs use color in an interactive terminal. To force color
when piping through a command that preserves ANSI escapes, pass
`--color always`; to keep logs plain, pass `--color never`.

The dataset is large. The downloader checks for at least `120GiB` of
free space before it starts because it needs room for downloaded split
parts and extraction. If the download stops after disk pressure or a
network interruption, free space and rerun the same command. The
underlying `gdown` call uses `--continue`, so completed and partial
parts can resume.

If you only have enough space to finish downloading the split archive
parts, but not enough to assemble and extract them yet, use:

```
./download_celeba_spoof.sh \
  --no-extract \
  --skip-space-check \
  --i-accept-non-commercial-research-terms
```

That resumes the Google Drive parts and stops before assembly,
extraction, and annotation validation.

If the split archive parts are already downloaded on the internal disk,
but the internal disk does not have enough free space for the combined
zip and the extracted dataset (roughly 2-3x the compressed size),
leave the parts where they are and send both the assembly and the
extraction to an external volume with `--output-dir`:

```
./download_celeba_spoof.sh \
  --skip-download \
  --output-dir /path/to/external-drive/celeba_spoof \
  --i-accept-non-commercial-research-terms
```

`--data-root` stays pointed at wherever the split parts already are;
`--output-dir` is where the combined zip is assembled, where files are
extracted, and where `validate_prepared_dataset` looks afterward. Point
training's `--data-root` at this same `--output-dir` once it finishes.
Only add `--skip-space-check` if you already know both volumes have
enough room; otherwise let the preflight check catch it before an
hours-long run gets partway through and dies on ENOSPC.

This command reads the existing `CelebA_Spoof.zip.001` ...
`CelebA_Spoof.zip.050` files from `ml/data/celeba_spoof/`, writes
`CelebA_Spoof.zip` to the external volume, and stops before extraction.
Replace `/path/to/external-drive/celeba_spoof` with the mounted path for
the external disk. It does not delete the downloaded split parts.

To download to a larger external volume:

```
./download_celeba_spoof.sh \
  --data-root /Volumes/big-disk/celeba_spoof \
  --i-accept-non-commercial-research-terms
```

Train and evaluate on CelebA-Spoof:

```
python src/train_cnn_celeba.py --epochs 15
python src/evaluate_cnn_celeba.py
```

Smoke run before the full training job:

```
python src/train_cnn_celeba.py --limit-train 1000 --limit-val 400 --epochs 1
python src/evaluate_cnn_celeba.py --limit 400
```

The CelebA-Spoof result is the decision point before Go/service work. If
the CNN fails to generalize beyond NUAA, serving it would only
productionize a dataset-specific model.
