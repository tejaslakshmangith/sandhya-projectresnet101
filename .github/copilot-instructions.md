# Copilot Instructions — SmartMine AI Safety Detection System

This project builds a SmartMine AI safety detection system using ResNet-101.

Tech stack:
- PyTorch ResNet-101 (pretrained ImageNet weights, transfer learning)
- FastAPI backend (POST /predict endpoint)
- Next.js 14 frontend (App Router)

Model specs:
- Input: 224x224 RGB images
- Output: class_name + confidence probability
- Classes: safe, unsafe, helmet, hazard
- Optimizer: Adam, lr=0.0001
- Batch size: 32, Epochs: 25
- Save path: ai-model/models/resnet101_smartmine.pth

Dataset format:
  ai-model/dataset/train/<class>/
  ai-model/dataset/val/<class>/
