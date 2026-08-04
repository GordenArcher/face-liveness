# Getting the NUAA dataset

This is not a one-click download, no automated fetch script exists for
it, the dataset owner requires a signed release agreement first:

1. Go to the [NUAA Imposter Database
   page](https://parnec.nuaa.edu.cn/_upload/tpl/02/db/731/template731/pages/xtan/NUAAImposterDB_download.html).
2. Download the release agreement, read it, sign it.
3. Email the signed scan to the address listed on that page
   (x.tan AT nuaa.edu.cn as of this writing, confirm on the actual page,
   contact details can change).
4. Wait for a response with access to the actual image data.

## Once you have it

Extract it so this directory looks like:

```
ml/data/raw/
  ClientFace/
    <subject-dir>/*.jpg   (live images, however the release organizes subjects)
  ImposterFace/
    <subject-dir>/*.jpg   (photo/print attack images)
```

`ClientFace/` and `ImposterFace/` are the two top-level folder names the
commonly redistributed "detected face" version of this dataset actually
uses. `dataset.py` recursively globs for images under each, so whatever
per-subject subdirectory structure the release actually has underneath
those two folders works without needing to flatten anything first.

If what you receive is organized differently (a single flat folder, a
different top-level naming), either rename/symlink to match the above,
or adjust `dataset.py`'s two folder names to match what you actually
got, this project has not independently verified every redistribution
of this dataset looks identical.

Not committed to this repo (see `.gitignore`): real biometric data, even
from a public research release, doesn't belong in git history.
