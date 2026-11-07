"""Image transforms shared by every torch dataset.

Keeping one set of transforms means a model comparison across datasets
is not silently confounded by different preprocessing.

The normalization stats are ImageNet's mean/std. Not because this model
is pretrained on ImageNet, it isn't, but because they're a reasonable,
well-tested default for natural face images rather than an arbitrary
choice, revisit if training instability shows up and this turns out to
matter.
"""

from torchvision import transforms

IMAGE_SIZE = 128

NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        # Small, realistic augmentations for a webcam-captured dataset,
        # not aggressive ones, NUAA's live/spoof distinction is a subtle
        # texture cue, augmentation strong enough to distort that cue
        # would hurt more than it helps.
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
    ]
)

EVAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD),
    ]
)
