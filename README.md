# EcomMMMU and MiEF

The folder contains the code for Towards Robust Multimodal E-Commerce Models: Strategic Verification of Visual Contributions from EcomMMMU Dataset.

## Introduction
We introduce [EcomMMMU](https://drive.google.com/file/d/1WmiGoAlEUJxezVMqwz6fJHSIx-175Q8r/view?usp=sharing), the first massive multitask multimodal understanding dataset for e-commerce to benchmark four essential capabilities. 
We also propose a data-centric multi-image e-commerce framework MiEF to facilitate the model's utilization of visual content and enhance its performance.

## Requirements

* python = 3.10.14
* torch = 2.4.1
* transformers = 4.46.0
* fire = 0.7.0
* scikit-learn = 1.5.2
* datasets = 3.0.1

## EcomMMMU Dataset

The dataset is available in [Google Drive](https://drive.google.com/file/d/1WmiGoAlEUJxezVMqwz6fJHSIx-175Q8r/view?usp=sharing).
EcomMMMU comprises 8 tasks centering on 4 essential e-commerce capabilities, including
shopping question perception, user behavior alignment, query-product perception, and shopping concept understanding
MMECInstruct is split into training sets, validation sets, and test sets. The visual_critical labels are involved for test samples.
