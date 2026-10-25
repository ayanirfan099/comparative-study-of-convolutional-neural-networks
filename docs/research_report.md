# Research report

Executed two seeded CPU synthetic runs (with and without augmentation), 100 training images, 20 validation, 20 test, 2 epochs. Non-augmented loss 2.2971; accuracy 0.200, training time 0.185s. These are **not CIFAR-10 results**. Actual JSON files include per-class metrics, latency, parameters, failure positions and full configuration.

![Synthetic learning curves](results/synthetic-no-augmentation/curves.png)
![Synthetic diagnostic Grad-CAM](results/synthetic-no-augmentation/gradcam.png)
