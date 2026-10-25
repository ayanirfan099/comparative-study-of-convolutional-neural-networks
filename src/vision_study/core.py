import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.247, 0.2435, 0.2616)
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(32, 10)

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)).flatten(1))


def build_model(kind="custom", pretrained=False, freeze=False):
    if kind not in {"custom", "resnet18"}:
        raise ValueError("Unknown architecture")
    if kind == "custom":
        if pretrained or freeze:
            raise ValueError("Pretrained/frozen flags apply only to ResNet")
        return SmallCNN()
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    if freeze and not pretrained:
        raise ValueError("Frozen transfer learning requires actual pretrained weights")
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model


def image_transform(augment=False, kind="custom", pretrained=False):
    # Pretrained ResNet gets ImageNet normalization and 224px inputs in train and eval.
    ops = [transforms.Resize((224, 224))] if kind == "resnet18" and pretrained else []
    if augment:
        ops += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(224 if ops else 32, padding=4),
        ]
    mean, std = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) if pretrained else (MEAN, STD)
    return transforms.Compose(ops + [transforms.ToTensor(), transforms.Normalize(mean, std)])


def loaders(
    root,
    seed=42,
    limit=1000,
    batch_size=32,
    augment=False,
    synthetic=False,
    kind="custom",
    pretrained=False,
):
    if limit < 20 or batch_size < 1:
        raise ValueError("limit >=20 and batch size >=1 required")
    train_tf = image_transform(augment, kind, pretrained)
    eval_tf = image_transform(False, kind, pretrained)
    if synthetic:
        train = datasets.FakeData(
            size=limit + max(20, limit // 5),
            image_size=(3, 32, 32),
            num_classes=10,
            transform=train_tf,
            random_offset=0,
        )
        val = datasets.FakeData(
            size=len(train),
            image_size=(3, 32, 32),
            num_classes=10,
            transform=eval_tf,
            random_offset=0,
        )
        test = datasets.FakeData(
            size=max(20, limit // 5),
            image_size=(3, 32, 32),
            num_classes=10,
            transform=eval_tf,
            random_offset=100000,
        )
    else:
        train = datasets.CIFAR10(root, train=True, download=True, transform=train_tf)
        val = datasets.CIFAR10(root, train=True, download=False, transform=eval_tf)
        test = datasets.CIFAR10(root, train=False, download=True, transform=eval_tf)
    order = np.random.default_rng(seed).permutation(len(train))
    n = min(limit, int(0.8 * len(train)))
    train_ids = order[:n]
    val_ids = order[n : n + min(max(20, n // 5), len(order) - n)]
    test_ids = np.random.default_rng(seed + 1).permutation(len(test))[
        : min(len(test), max(20, n // 5))
    ]
    generator = torch.Generator().manual_seed(seed)
    return (
        DataLoader(
            Subset(train, train_ids), batch_size=batch_size, shuffle=True, generator=generator
        ),
        DataLoader(Subset(val, val_ids), batch_size=batch_size),
        DataLoader(Subset(test, test_ids), batch_size=batch_size),
    ), {"train": train_ids.tolist(), "validation": val_ids.tolist(), "test": test_ids.tolist()}


def epoch(model, loader, optimizer=None, freeze=False):
    model.train(optimizer is not None)
    if freeze and optimizer is not None:
        # Frozen backbone includes BatchNorm running statistics, not just parameters.
        for name, child in model.named_children():
            if name != "fc":
                child.eval()
    total = correct = 0
    loss_sum = 0.0
    ys = []
    predictions = []
    for x, y in loader:
        with torch.set_grad_enabled(optimizer is not None):
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        loss_sum += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        total += len(y)
        ys.extend(y.tolist())
        predictions.extend(logits.argmax(1).tolist())
    return {"loss": loss_sum / total, "accuracy": correct / total}, ys, predictions


def gradcam(model, x, target_layer):
    activation = []
    gradient = []

    def hook(module, inputs, output):
        activation.append(output)
        output.register_hook(lambda grad: gradient.append(grad))

    handle = target_layer.register_forward_hook(hook)
    try:
        model.eval()
        model.zero_grad()
        x = x.detach().clone().requires_grad_(True)
        logits = model(x)
        logits[0, logits[0].argmax()].backward()
        cam = (activation[0] * gradient[0].mean(dim=(2, 3), keepdim=True)).sum(1).relu()
        cam = nn.functional.interpolate(
            cam[:, None], size=x.shape[-2:], mode="bilinear", align_corners=False
        )[0, 0]
        cam = cam / (cam.max() + 1e-12)
        return cam.detach().numpy()
    finally:
        handle.remove()


def load_checkpoint(path):
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    model = build_model(checkpoint["kind"], pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint
