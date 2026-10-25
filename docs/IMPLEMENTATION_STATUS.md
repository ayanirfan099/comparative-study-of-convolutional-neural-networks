# Implementation status

Implemented but not fully benchmarked

A reproducible PyTorch training/evaluation system for a small CNN and ResNet18, including explicit pretrained-weight provenance, augmentation, checkpoint selection, Grad-CAM and a checkpoint-backed Streamlit demo.

Four tests passed: architecture outputs, Grad-CAM finite maps, invalid transfer flags, disjoint train/validation IDs, seeded repeated-run equality, checkpoint reload, end-to-end metric/figure outputs and CLI help. Initial Streamlit page checked with a real synthetic-trained checkpoint.

Official CIFAR-10 download failed with proxy tunnel 403 at www.cs.toronto.edu. No real-data CIFAR-10 or pretrained transfer comparison has executed. Synthetic images have independent random labels: accuracy is a software diagnostic, not scientific recognition performance. Pretrained weights must download successfully; no silent random-weight fallback. No GPU benchmark, confidence calibration or full hyperparameter search. Upload browser interaction remains untested. Grad-CAM is a qualitative diagnostic, not causal explanation.

Exact commands: [reproducibility](reproducibility.md). Push status and commit identity are recorded in the portfolio summary; remote CI success is not implied.


Additional verified check: official torchvision ImageNet ResNet18 weights downloaded successfully from download.pytorch.org. A frozen-backbone one-epoch training/evaluation/Grad-CAM/checkpoint run completed on 20 synthetic training images; see results/pretrained-transfer-smoke/metrics.json. This validates the transfer execution path, not CIFAR-10 accuracy. The custom two-run synthetic experiment and all real CIFAR-10 limitations above remain unchanged.
