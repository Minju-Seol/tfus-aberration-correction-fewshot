# Few-shot Phase-Amplitude Aberration Correction for Phased Array Transducer in Real-time for Transcranial Focused Ultrasound

This repository contains the implementation of a few-shot deep surrogate framework that jointly predicts the phase and amplitude of each element of a 96-element phased-array transducer in real time, for skull-induced aberration correction in transcranial focused ultrasound (tFUS).

*Anonymous submission for double-blind review.*

## Overview

Transcranial focused ultrasound requires patient-specific phase and amplitude correction to compensate for skull-induced acoustic distortion. Conventional time-reversal (TR) simulation provides accurate correction but relies on computationally expensive full-wave solvers, making it unsuitable for real-time or iterative treatment planning.

This work proposes a geometry-aware deep surrogate model that:
- Encodes skull geometry along the acoustic path between each transducer element and the focal target
- Jointly predicts per-element phase (via circular classification) and amplitude (via regression)
- Is pretrained across diverse skull geometries and fine-tuned with only 10 target points on an unseen skull
- Achieves real-time inference of the full 96-element steering profile

## Features

- **Geometry Feature Encoder**: combines a 3D convolutional skull patch embedding, Fourier-embedded transducer/target positions, and skull-interface intersection distances into a shared geometry representation
- **Phase Classification Model**: predicts phase via circular soft-label classification with circular expectation decoding, avoiding the discontinuity of direct angular regression
- **Amplitude Regression Model**: predicts per-element amplitude with Huber loss on standardized targets
- **Few-shot Fine-tuning**: adapts the pretrained model to a new, unseen skull using only 10 target points, with the 3D skull encoder frozen during fine-tuning
- **Leave-one-skull-out (LOO) evaluation** across all subjects, reporting phase/amplitude error and inference time against TR simulation ground truth

## Repository Structure

```
.
├── models/
│   ├── Models_archive.py        # Phase and Amplitude model architectures
│   └── __init__.py
├── data/
│   └── sample_inference_data.pt # Example Data for model implementation
├── dataset.py                   # Dataset classes for amplitude and phase prediction
├── utils.py                     # Geometry computation and pre/post-processing utilities
├── fewshot_amp_training.py      # LOO training + few-shot fine-tuning (amplitude)
├── fewshot_phase_training.py    # LOO training + few-shot fine-tuning (phase)
├── Implementation.py            # Inference script using fine-tuned checkpoints
└── README.md
```

## Example Dataset

A small example dataset is provided to illustrate the expected data format and allow the training/inference scripts to be run end-to-end without requiring full patient CT data.

- `data/inference_data.pt` — example tensor dictionary containing the keys: `skull`, `target`, `td`, `amp`, `phase`
- This example data is for demonstration of the pipeline only and does not reflect the full-scale dataset used in the paper.

## Requirements

- Python 3.12
- PyTorch
- NumPy

To run training:
```bash
python fewshot_amp_training.py --data_dir ./data --result_dir ./results
python fewshot_phase_training.py --data_dir ./data --result_dir ./results
```

To run inference with fine-tuned checkpoints:
```bash
python Implementation.py --data_dir ./data --model_dir ./results --result_dir ./results
```
