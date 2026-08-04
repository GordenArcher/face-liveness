# Getting the NUAA dataset

The [NUAA Imposter Database
page](https://parnec.nuaa.edu.cn/_upload/tpl/02/db/731/template731/pages/xtan/NUAAImposterDB_download.html)
provides three direct downloads, all hosted on Google Drive, no
agreement or email required:

1. Raw, unprocessed images, 390M zip
2. Face-detector output (cropped face images), 73M zip
3. Geometrically normalized images, 56M zip

Use **format 2**, the face-detector output. That's the version this
project's `dataset.py` is built around, it's already cropped to faces
rather than full original photos, which is what both the LBP baseline
and later models actually need as input.

No automated fetch script exists for this in the repo, the files are on
Google Drive rather than a plain HTTP host a script could pull from
directly, so downloading through the browser is the practical path.

## Once you have it

Extract the zip so this directory looks like:

```
ml/data/raw/
  ClientFace/
    <subject-dir>/*.jpg   (live images)
  ImposterFace/
    <subject-dir>/*.jpg   (photo/print attack images)
```

`ClientFace/` and `ImposterFace/` are the two top-level folder names the
format 2 zip actually uses. `dataset.py` recursively globs for images
under each, so whatever per-subject subdirectory structure exists
underneath those two folders works without needing to flatten anything
first.

If your extracted structure looks different from the above, either
rename/symlink to match it, or adjust the two folder names in
`dataset.py` to match what you actually got.

Not committed to this repo (see `.gitignore`): real biometric data, even
from a public research release, doesn't belong in git history.
