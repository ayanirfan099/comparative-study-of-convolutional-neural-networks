import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

from .core import CLASSES, build_model, epoch, gradcam, loaders, seed_all


def run(
    output="results/cpu",
    data="data",
    seed=42,
    epochs=2,
    limit=1000,
    batch_size=32,
    kind="custom",
    augment=False,
    pretrained=False,
    freeze=False,
    synthetic=False,
    patience=3,
):
    if epochs < 1 or patience < 1:
        raise ValueError("Epochs and patience must be positive")
    seed_all(seed)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    config = locals().copy()
    config = {k: v for k, v in config.items() if k not in {"out", "config"}}
    model = build_model(kind, pretrained, freeze)
    (train, val, test), indices = loaders(
        data, seed, limit, batch_size, augment, synthetic, kind, pretrained
    )
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=1)
    best = float("inf")
    stale = 0
    history = []
    start = perf_counter()
    for step in range(epochs):
        tr, _, _ = epoch(model, train, opt, freeze)
        va, _, _ = epoch(model, val)
        history.append(
            {
                "epoch": step + 1,
                "train": tr,
                "validation": va,
                "learning_rate": opt.param_groups[0]["lr"],
            }
        )
        scheduler.step(va["loss"])
        if va["loss"] < best:
            best = va["loss"]
            stale = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "kind": kind,
                    "pretrained": pretrained,
                    "classes": CLASSES,
                    "config": config,
                },
                out / "best.pt",
            )
        else:
            stale += 1
        if stale >= patience:
            break
    training_seconds = perf_counter() - start
    checkpoint = torch.load(out / "best.pt", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    metrics, y, pred = epoch(model, test)
    x, _ = next(iter(test))
    model.eval()
    with torch.inference_mode():
        model(x[:1])
        start = perf_counter()
        for _ in range(20):
            model(x[:1])
    latency = (perf_counter() - start) / 20
    result = {
        "data": "synthetic FakeData software smoke only"
        if synthetic
        else "CIFAR-10 official test subset",
        "config": config,
        "history": history,
        "test": metrics,
        "classification_report": classification_report(
            y, pred, labels=list(range(10)), target_names=CLASSES, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y, pred, labels=list(range(10))).tolist(),
        "training_seconds": training_seconds,
        "single_image_latency_seconds": latency,
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "test_count": len(y),
        "failure_examples": [
            {"test_position": i, "true": a, "predicted": b}
            for i, (a, b) in enumerate(zip(y, pred, strict=True))
            if a != b
        ][:20],
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    (out / "splits.json").write_text(json.dumps(indices))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot([h["train"]["loss"] for h in history], label="train")
    ax.plot([h["validation"]["loss"] for h in history], label="validation")
    ax.legend()
    ax.set(xlabel="Epoch (0-based)", ylabel="Cross entropy")
    fig.savefig(out / "curves.png")
    plt.close(fig)
    layer = model.features[3] if kind == "custom" else model.layer4[-1].conv2
    cam = gradcam(model, x[:1], layer)
    fig, ax = plt.subplots()
    image = x[0].permute(1, 2, 0).numpy()
    image = (image - image.min()) / (np.ptp(image) + 1e-12)
    ax.imshow(image)
    ax.imshow(cam, alpha=0.5, cmap="jet")
    ax.axis("off")
    ax.set_title("Predicted-class Grad-CAM, held-out image")
    fig.savefig(out / "gradcam.png")
    plt.close(fig)
    return result


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="results/cpu")
    p.add_argument("--data", default="data")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--limit", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--kind", choices=["custom", "resnet18"], default="custom")
    for flag in ["augment", "pretrained", "freeze", "synthetic"]:
        p.add_argument("--" + flag, action="store_true")
    print(json.dumps(run(**vars(p.parse_args())), indent=2))


if __name__ == "__main__":
    main()
