# Study guide

These notes explain implemented methods; do not claim independent authorship or experiments beyond the recorded results.

## What must remain frozen in a frozen feature extractor?

Weights and BatchNorm running statistics. The implementation sets backbone modules to evaluation mode during head training.

## Is pretrained ResNet versus a scratch CNN a fair architecture-only comparison?

No. Pretraining data, resolution, normalization and compute differ; report it as a transfer-learning comparison with those confounders.

## What can Grad-CAM establish?

It visualizes weighted feature-map gradients for a selected class; it is a qualitative localization diagnostic, not proof of causal reasoning.
