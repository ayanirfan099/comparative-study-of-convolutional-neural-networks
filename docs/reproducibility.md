# Reproduce

Python >=3.11; executed in Python 3.12 on CPU. From repository root:
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m vision_study.cli --data data --limit 1000 --epochs 2
python -m vision_study.cli --data data --limit 1000 --epochs 2 --augment --output results/augmented
python -m vision_study.cli --kind resnet18 --pretrained --freeze --limit 1000 --epochs 2 --output results/frozen
python -m vision_study.cli --kind resnet18 --pretrained --limit 1000 --epochs 2 --output results/finetune
# Offline software test:
python -m vision_study.cli --synthetic --limit 100 --epochs 2 --output results/synthetic-no-augmentation
streamlit run src/vision_study/app.py
pytest -q
ruff check .
ruff format --check .
```

Use CPU PyTorch wheels from https://download.pytorch.org/whl/cpu if installing the deep learning dependencies on a CPU machine. Remote datasets and weights require separate access. Do not load untrusted pickle/joblib/checkpoint artifacts.
