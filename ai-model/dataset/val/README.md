# Validation Data — Placeholder

Place validation images here, organised by class:

```
val/
├── safe/
│   └── ...
├── unsafe/
│   └── ...
├── helmet/
│   └── ...
└── hazard/
    └── ...
```

Each sub-folder name becomes the class label used by `torchvision.datasets.ImageFolder`.
