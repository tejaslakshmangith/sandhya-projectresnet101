"""ResNet-101 model definition for SmartMine AI Safety Detection."""

import torch
import torch.nn as nn
import torchvision.models as models


class SmartMineResNet101(nn.Module):
    """ResNet-101 with a custom classification head for mine-safety detection.

    Pretrained ImageNet weights are used for transfer learning. All layers
    except ``layer3``, ``layer4``, and the custom ``fc`` head are frozen.

    Args:
        num_classes: Number of output classes (default: 4).
    """

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.num_classes = num_classes

        # Load pretrained backbone
        backbone = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)

        # Freeze all backbone layers initially
        for param in backbone.parameters():
            param.requires_grad = False

        # Unfreeze layer3, layer4 for fine-tuning
        for param in backbone.layer3.parameters():
            param.requires_grad = True
        for param in backbone.layer4.parameters():
            param.requires_grad = True

        # Keep all backbone modules except the original fc
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

        # Custom classification head: 2048 -> 512 -> num_classes
        self.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def __repr__(self) -> str:  # noqa: D105
        return (
            f"SmartMineResNet101(\n"
            f"  backbone=ResNet-101 (pretrained, layer3+layer4 unfrozen)\n"
            f"  fc=Linear(2048→512) → ReLU → Dropout(0.4) → Linear(512→{self.num_classes})\n"
            f"  num_classes={self.num_classes}\n"
            f")"
        )


def get_model(num_classes: int = 4, device: str = "cpu") -> SmartMineResNet101:
    """Factory function that returns a ``SmartMineResNet101`` on *device*.

    Args:
        num_classes: Number of output classes.
        device: PyTorch device string, e.g. ``"cuda"`` or ``"cpu"``.

    Returns:
        Model moved to the requested device.
    """
    model = SmartMineResNet101(num_classes=num_classes)
    model = model.to(device)
    return model
