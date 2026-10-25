from pathlib import Path

import streamlit as st
import torch
from PIL import Image

from vision_study.core import image_transform, load_checkpoint

st.title("CIFAR-10 research classifier")
st.caption(
    "Scores may be poorly calibrated. Synthetic smoke checkpoints have no useful recognition ability. Load only trusted local checkpoints."
)
p = Path(st.text_input("Checkpoint", "results/cpu/best.pt"))
if not p.exists():
    st.info("Run training first.")
    st.stop()
model, meta = load_checkpoint(p)
st.json(meta["config"])
f = st.file_uploader("Image", type=["png", "jpg", "jpeg"])
if f:
    try:
        if f.size > 5 * 1024 * 1024:
            raise ValueError("Maximum upload 5 MiB")
        image = Image.open(f)
        image.load()
        image = image.convert("RGB")
        st.image(image)
        x = image_transform(False, meta["kind"], meta["pretrained"])(image)[None]
        with torch.inference_mode():
            scores = model(x).softmax(1)[0]
        st.bar_chart(dict(zip(meta["classes"], scores.tolist(), strict=True)))
    except (ValueError, OSError) as exc:
        st.error(str(exc))
