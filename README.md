# Aerial Object Detection with Edge Deployment

Exploring the tradeoffs between accuracy and inference speed when deploying transformer-based object detectors on resource-constrained hardware. Built using VisDrone dataset as a proxy for drone perception tasks.

## Why This Project

I wanted to understand what actually breaks when you try to run modern detection models under edge constraints. Everyone says "just quantize it" but I wanted to see the real accuracy/latency tradeoffs myself, especially for small objects which are notoriously hard in aerial imagery.

## What I'm Using

- **Dataset:** VisDrone-DET (drone footage, 10 object classes)
- **Model:** RT-DETR (transformer-based detector)
- **Optimization:** FP32 → FP16 → INT8 quantization pipeline
- **Export:** ONNX for deployment

## Results

*TODO: Fill in after experiments*

| Variant | mAP@0.5 | AP_small | Latency (ms) | Size (MB) |
|---------|---------|----------|--------------|-----------|
| FP32 baseline | | | | |
| FP16 | | | | |
| INT8 PTQ | | | | |

### Accuracy vs Latency

*TODO: Add Pareto frontier plot*

## Key Findings

*TODO: What actually surprised me, what broke, what worked better than expected*

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/aerial-detection-edge.git
cd aerial-detection-edge
pip install -r requirements.txt
```

### Data Setup

1. Download VisDrone-DET from [here](https://github.com/VisDrone/VisDrone-Dataset)
2. Unzip into `data/raw/`
3. Run conversion: `python src/data/prepare.py`

## Project Structure

```
├── data/
│   ├── raw/           # Original VisDrone downloads
│   └── processed/     # YOLO-format annotations
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_training.ipynb
│   ├── 03_quantization.ipynb
│   └── 04_benchmarking.ipynb
├── src/
│   ├── data/          # Dataset prep scripts
│   ├── train/         # Training code
│   ├── optimize/      # Quantization, export
│   └── evaluate/      # Benchmarking, visualization
├── results/
│   ├── figures/
│   └── checkpoints/
└── configs/
```

## Reproduction

*TODO: Add commands to reproduce key results*

## What I'd Do With More Time

- Quantization-aware training to recover INT8 accuracy
- Knowledge distillation from larger model
- Test on actual Jetson Nano hardware
- Multi-scale inference for small objects

## References

- [VisDrone Dataset](https://github.com/VisDrone/VisDrone-Dataset)
- [RT-DETR Paper](https://arxiv.org/abs/2304.08069)
- [Ultralytics Docs](https://docs.ultralytics.com/)
