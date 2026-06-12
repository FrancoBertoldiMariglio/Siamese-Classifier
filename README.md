# Siamese-Classifier

A **Siamese network** in PyTorch for image similarity verification, applied to **identity document (DNI) validation**: given two images, the network outputs the probability that they belong to the same class.

Based on [sohaib023/siamese-pytorch](https://github.com/sohaib023/siamese-pytorch), adapted and deployed for a private document-verification dataset.

## Approach

- Pre-trained `torchvision.models` backbones as feature extractors
- Sigmoid output head → **binary classification** (same / different) with BCE loss, instead of the more common triplet loss
- Element-wise multiplication of the two embeddings before the classification head
- Dropout + BatchNorm in the head

## Training & evaluation

```bash
conda env create -f environment.yml && conda activate pytorch_siamese

python train.py --train_path data/train --val_path data/val --out_path runs/
python eval.py  --val_path data/val --checkpoint runs/best.pth
```

Dataset format: one folder per class, all class folders under a common root. Training logs to TensorBoard.

## Deployment: ONNX → TensorRT

The repo includes the full export pipeline for optimized inference:

```bash
python torch_to_onnx.py   # PyTorch checkpoint → ONNX
python onnx_to_trt.py     # ONNX → TensorRT engine
python infer_tensorrt.py  # Inference with the TensorRT engine
```

> Note: the ID-document dataset is private and not included in this repository.
