# Training Data — Placeholder

Place training images here, organised by class:

```
train/
├── safe/
│   ├── image1.jpg
│   └── ...
├── unsafe/
│   └── ...
├── helmet/
│   └── ...
└── hazard/
    └── ...
```

Each sub-folder name becomes the class label used by `torchvision.datasets.ImageFolder`.
