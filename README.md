# EcomMMMU and SUMEI

The folder contains the code for EcomMMMU: Strategic Utilization of Visuals for Robust Multimodal E-Commerce Models.

## Introduction
We introduce [EcomMMMU](https://drive.google.com/file/d/1WmiGoAlEUJxezVMqwz6fJHSIx-175Q8r/view?usp=sharing), the first large-scale multimodal multitask understanding dataset for e-commerce to benchmark four essential capabilities. 
We also propose a method, SUMEI, to facilitate the model's utilization of visual content and enhance its performance.

## Requirements

* python = 3.10.14
* torch = 2.4.1
* transformers = 4.49.0
* scikit-learn = 1.5.2
* datasets = 3.0.1

## EcomMMMU Dataset

The dataset is available in [Google Drive](https://drive.google.com/file/d/1WmiGoAlEUJxezVMqwz6fJHSIx-175Q8r/view?usp=sharing).
EcomMMMU comprises 8 tasks centering on 4 essential e-commerce capabilities, including
shopping question perception, user behavior alignment, query-product perception, and shopping concept understanding
EcomMMMU is split into training sets, validation sets, and test sets. The visual_critical labels are involved for test samples.


## Visual Utility Assessment
To generate pseudo-labels for product images, run <code>python pseudo_labeling.py</code>.
 <!-- By default we use [ec-llava](https://huggingface.co/meta-llama/ec-llava) as the labeling model. -->

The generated pseudo-labeled data will be stored in <code>data/verification</code> for training the instance-level image verifier.

## Visual Utility Predictor Finetuning

To finetune the verifier, run <code>./finetune.sh $stage</code>.

<code>$stage</code> indicates the finetuning stage and should be set as <code>verification</code> in this procedure.

Example:
```
./finetune.sh verification
```

The finetuning code is derived from [lmms-finetune](https://github.com/zjysteven/lmms-finetune).

## Visual Utility Prediction
To verify the contribution of product images, run <code>python verification.py</code>.

The generated contribution label data for conducting downstream task will be stored in <code>data/downstream</code> for finetuning and inference the downstream multimodal large language models.


## Downstream Model Finetuning

To finetune the model to conduct downstream tasks, run <code>./finetune.sh $stage</code>.

<code>$stage</code> indicates the finetuning stage and should be set as <code>downstream</code> in this procedure.

Example:
```
./finetune.sh downstream
```

## Task Inference
To conduct inference, run <code>python inference.py --task $task</code>.

<code>$task</code> specifies the task to be tested.

Example:
```
python inference.py --task answerability_prediction
```

## Evaluation
To evaluate the results on specific task, run <code>python evaluate.py --task $task</code>.

<code>$task</code> is the task on which to conduct the evaluation.

Example:
```
python evaluate.py --task answerability_prediction
```
