# face-liveness

Face verification that can tell the difference between a live person
and someone trying to get past it with a photo, a printed picture, or a
phone/screen held up to the camera.

## The problem

"Face verification" sounds like one problem. It's two, and they're not
equally hard:

**Face recognition**: is this face the same person as that other face.
A pretrained embedding model turns a detected, aligned face into a
vector, and comparing two faces becomes a cosine similarity check
against a threshold. This part is largely solved territory; there's
integration work here, not much open research.

**Liveness detection** (presentation attack detection, PAD in the
literature): is the thing in front of the camera an actual live human,
not a printed photo, not a screen replaying a video, not a mask. This is
the hard, actively-researched half, and it's the part this project
spends its effort on.

## How the pipeline works

```
capture → detect face → LIVENESS CHECK → (if live) → embedding → compare
                              ↓
                        (if spoof) → reject, recognition never runs
```

Liveness detection gates recognition, not the other way around. A photo
held up to the camera gets rejected before face matching ever sees it.
This ordering is the actual fraud prevention; recognition and liveness
aren't independent features, one blocks the other.

## Architecture

```mermaid
flowchart LR
    subgraph GoClient["go-client (CLI + gateway)"]
        CLI["register-face / verify-face\nidentify-face"]
        Enc["internal/crypto\nAES-256-GCM behind a KeyProvider interface"]
        Store["internal/storage"]
        CLI --> Enc --> Store
    end

    PG[("Postgres\npeople + face_embeddings\n(ciphertext only)")]
    Store <--> PG

    CLI <-->|"mTLS gRPC"| MLService

    subgraph MLService["face-service (Python, stateless)"]
        Detect["face detection + alignment"]
        Live["liveness classifier\n(LBP+SVM, then CNN)"]
        Embed["embedding model\n(FaceNet / ArcFace)"]
        Detect --> Live
        Live -->|"live"| Embed
        Live -->|"spoof"| Reject["reject, no embedding computed"]
    end
```

One RPC does detect → liveness check → embedding in a single pass (the
face crop is reused for both the liveness classifier and the embedding
model), returning either a rejection reason or an embedding plus a
liveness confidence score.

Go owns storage, encryption, and orchestration. The Python service is
stateless: it never touches a database, never sees a name or date of
birth, only image bytes and the scores/embeddings it computes from
them. That split is deliberate: it means the matching/liveness models
can be retrained, swapped, or rebenchmarked without touching how
identities are stored, and it means a compromise of the ML service
alone doesn't expose biographic data.

## Metrics

Evaluated on APCER, BPCER, and ACER, not just accuracy:

- **APCER**: attack presentation classification error rate: how often
  a spoof fools the model into thinking it's live. The dangerous
  failure mode.
- **BPCER**: bona fide presentation classification error rate: how
  often a real person gets wrongly rejected. The annoying but safe
  failure mode.
- **ACER**: the average of the two, the headline number.

Accuracy alone hides which of these is happening. A model that always
predicts "live" can post deceptively high accuracy on an imbalanced test
set while its APCER sits at 100%.

## Status

| Milestone                                  | Status             | Notes                                                                                                                                                                                                |
| ------------------------------------------ | ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| M1: LBP + SVM baseline                     | done, single split | ACER 0.3300 (APCER 0.0419, BPCER 0.6181) on the corrected subject-disjoint split, 843 test images. High BPCER given how few test subjects there are, see k-fold below for a more robust estimate     |
| M2: small CNN from scratch                 | done, single split | ACER 0.2216 (APCER 0.0311, BPCER 0.4121), beats M1 on the same split, but the single-split validation signal used for epoch selection turned out to be unreliable, see "A real bug" and k-fold below |
| M2b: k-fold cross-validation               | in progress        | addresses the single-split validation noise directly, see "Cross-validation" below                                                                                                                   |
| M3: generalization stretch (CelebA-Spoof)  | not started        | tests whether M2 overfits to NUAA's narrow capture conditions                                                                                                                                        |
| M4: export to ONNX, serve                  | not started        |                                                                                                                                                                                                      |
| M5: Go client + encrypted Postgres storage | not started        |                                                                                                                                                                                                      |
| M6: mTLS between services                  | not started        |                                                                                                                                                                                                      |

Sequenced deliberately: classical technique first, so there's a real
number to compare a neural net against, then a CNN, then a harder
dataset to find out where that CNN's assumptions break.

### A real bug: subject leakage in the original split

The first version of `dataset.py` split at the image level, stratified
by label only. NUAA numbers subjects identically across `ClientFace/`
and `ImposterFace/` (subject `0007`'s live photos and photos of a spoof
attack against them are both under a folder named `0007`), and with
only a few dozen subjects total and thousands of images per subject
across multiple sessions, an image-level split put the same subject's
photos in both train and test.

That let a model partly learn "do I recognize this specific person"
instead of the actual live-vs-spoof texture cue, a shortcut that
doesn't exist at real verification time against someone the model has
never seen. It's why the CNN's validation ACER hit exactly 0.0000: not
a good model, a leaking one.

`dataset.py` now splits by subject, no subject's images appear in more
than one split. NUAA's total subject count is small (low double digits),
so this produces a genuinely small number of test subjects, that's an
honest limitation of the dataset itself, not something to hide by going
back to a split that would report a better-looking, meaningless number.
Both M1 and M2 need to be retrained and re-evaluated against this
corrected split before either result means anything.

### A second real bug: noisy single-split validation

Rerunning M2 on the corrected subject-disjoint split, `train_cnn.py`
selected its best checkpoint at validation ACER 0.0007, epoch 12 of 15.
The same model scored 0.2216 ACER on the real held-out test set, a huge
gap.

With only a handful of subjects in the validation set, val ACER is a
noisy, high-variance number, it swings based on which specific
subject's images happen to be easy or hard that particular epoch, not
purely on genuine model quality. Picking "the single best epoch out of
15" is itself a form of overfitting to that noise: epoch 12 wasn't
necessarily a better model, it may just have been the epoch that got
lucky on a tiny validation set. The test result, not the validation
result, was the trustworthy number.

### Cross-validation

`kfold.py`, `cross_validate_baseline.py`, and `cross_validate_cnn.py`
address this directly with subject-disjoint k-fold cross-validation,
via sklearn's `GroupKFold` (group = subject_id). Every subject serves as
test data in exactly one fold across the k runs, so the whole dataset
gets used for evaluation instead of permanently reserving a slice of an
already-small subject pool as a holdout that never contributes to
training. Reporting mean and standard deviation across folds is a
meaningfully more robust estimate than trusting a single split, and the
standard deviation itself is informative: small means the estimate is
stable, large is an honest signal it still shouldn't be trusted to one
decimal place.

Both CV scripts call `kfold.make_folds` with the same `n_splits`, and
`GroupKFold`'s fold assignment is deterministic for a given input order,
so the two models are evaluated on identical folds without either
script needing to explicitly coordinate with the other.

```
python src/cross_validate_baseline.py
python src/cross_validate_cnn.py
```

`cross_validate_cnn.py` trains a fresh model per fold (5 by default),
expect roughly 5x the runtime of a single `train_cnn.py` run.

## Getting started

This project is pinned to Python 3.12. The repo includes a
`.python-version` for pyenv, and the commands below create an isolated
venv so the scripts do not accidentally run against a system Python
with missing or mismatched scientific packages.

### M1: LBP + SVM baseline

```
cd ml
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Get the dataset first, see `ml/data/README.md` for which of the three
available formats to use.

```
python src/train_baseline.py
python src/evaluate.py
```

`train_baseline.py` writes the exact train/val/test split it used to
`ml/models/split.json`. `evaluate.py` reads that back instead of
recomputing its own split, so the reported numbers are guaranteed to be
against data the model never saw during training. If it recomputed the
split independently, a later change to the splitting logic could
silently produce a different held-out set than the one training
actually used, and the reported numbers would be meaningless without
anyone noticing.

### M2: CNN trained from scratch

Same split as M1, `train_cnn.py` reads `ml/models/split.json` directly
rather than computing its own, so the comparison against M1's numbers
is on identical data.

```
python src/train_cnn.py
python src/evaluate_cnn.py
```

`train_cnn.py` selects the best checkpoint by validation ACER, not
validation loss, loss is what the optimizer minimizes but it isn't the
number this project reports, the lowest-loss checkpoint and the
lowest-ACER checkpoint aren't guaranteed to be the same epoch.

## Datasets

- **[NUAA Photograph Imposter
  Database](https://parnec.nuaa.edu.cn/_upload/tpl/02/db/731/template731/pages/xtan/NUAAImposterDB_download.html)**,
  small, print-attack-only. Direct download via Google Drive, no
  agreement required. Three formats are offered, use the
  face-detector-output one, see `ml/data/README.md` for details. Used
  for M1/M2.
- **[CelebA-Spoof](https://github.com/ZhangYuanhan-AI/CelebA-Spoof)**,
  larger, covers print, replay, and mask attacks. Distributed via
  Google Drive/Baidu links, sometimes behind a request form. Used for
  M3.

## Tech stack

- **Python**: face detection/alignment, liveness classifier
  (scikit-learn for M1, PyTorch for M2+), embedding model, gRPC serving
- **Go**: CLI, storage, encryption, orchestration
- **Postgres**: encrypted face embeddings, kept in a separate table
  from biographic data
- **gRPC + mTLS**: service boundary and transport auth between Go and
  Python

## Out of scope

- **Depth/IR-based liveness.** Commercial PAD systems lean on depth/IR
  cameras precisely because RGB-only liveness detection has real, known
  limits, a good enough printed photo or replayed video can still fool
  a single-image classifier. This project is RGB-only by necessity
  (webcam/phone camera); that limitation is stated here rather than
  papered over.
- **Active liveness** (blink/head-turn challenges). Passive, single-image
  liveness only.
- **Multi-modal fusion** (RGB + depth + IR combined), the realistic way
  commercial systems close the gap a single RGB camera can't, and out of
  reach without that hardware.
