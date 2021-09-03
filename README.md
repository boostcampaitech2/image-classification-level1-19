# image-classification-level1-19
image-classification-level1-19 created by GitHub Classroom

P-Stage level-1 image classification competition. (shape of wearing mask, gender, age for a person)<br>
19조 (BCAA) Solution

# Archive contents
```
input/
├── data/
│ ├── eval - evaluation dataset
│ └── train - train dataset
code/
├── train.py
├── inference.py
├── dataset.py
├── evaluation.py
├── loss.py
├── model.py
└── model
  └── exp1/
```
- `input/data/eval`: evaluation dataset
- `input/data/train`: train dataset
- `code/train.py`: main script to start training
- `code/inference.py`: evaluation of trained model
- `code/dataset.py`: custom data loader for dataset
- `code/evaluation.py`: function to score, matric
- `code/loss.py`: contains loss functions
- `code/model.py`: contains custom or pre-trained model
- `model/`: trained models are saved here like(exp1, exp2,...)

# Requirements
- Linux version 4.4.0-59-generic
- Python >= 3.8.5
- PyTorch >= 1.7.1

`pip install -r requirements.txt` : install the necessary packages.

### Hadware
- CPU: Intel(R) Xeon(R) Gold 5220 CPU @ 2.20GHz
- GPU: Tesla V100-SXM2-32GB

# Training
- general training with default args ```python train.py```
- K-Fold training with default args ```python train.py --k_fold```
- singel model training ```python train.py --single```
- multiple model training ```python train.py --single --k_fold```

# Inference
```python inference.py --model_dir {'MODEL_PATH'}```
ex. <br>
`python inference.py --model_dir "./models/exp1"`
