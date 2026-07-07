# 📱 UdaciSense: Optimized Object Recognition

This project presents a complete **model compression pipeline** for a computer vision system designed to recognize household objects. The model has been optimized for **mobile deployment**, achieving improvements in size and inference speed while preserving accuracy.

---

## 🚀 Overview

The objective of this project is to transform a pre-trained model into an efficient, production-ready solution suitable for **resource-constrained environments** such as smartphones and edge devices.

Key outcomes:

* Reduced model size
* Faster inference time
* Maintained high accuracy

---

## ⚙️ Installation

1. Clone the repository:

```sh
git clone https://github.com/eljandoubi/Optimized_Mobile_Object_Recognition.git
cd Optimized_Mobile_Object_Recognition
```

2. Install dependencies using **uv**:

```sh
uv sync
```

3. Install the project as a local package:

```sh
uv pip install -e .
```

---

## 🏗️ Project Structure

```
├── notebooks/
│   ├── 01_baseline.ipynb
│   ├── 02_compression.ipynb
│   ├── 03_pipeline.ipynb
│   └── 04_deployment.ipynb
│
├── compression/
│   ├── in-training/
│   │   ├── distillation.py
│   │   ├── gradual_pruning.py
│   │   └── quantization_aware.py
│   └── post-training/
│       ├── graph_optimization.py
│       ├── pruning.py
│       └── quantization.py
│
├── models/        # Saved models (baseline & optimized)
├── results/       # Evaluation outputs
│
├── utils/
│   ├── compression.py
│   ├── data_loader.py
│   ├── evaluation.py
│   ├── model.py
│   └── visualization.py
│
├── README.md
├── requirements.txt
├── report.md
└── setup.py
```

---

## 📊 Results

The optimized model achieves:

* ✔️ ~30% reduction in model size
* ✔️ ~40% reduction in inference time
* ✔️ Accuracy maintained within ~5% of baseline

---

## 🧠 Compression Techniques Used

* **Pruning** – Removing redundant weights
* **Quantization** – Lower precision representation for efficiency
* **Knowledge Distillation** – Transferring knowledge from a larger model
* **Graph Optimization** – Improving execution efficiency

These techniques are combined into a **multi-stage compression pipeline** for optimal performance.

---

## 📱 Deployment

The final model is prepared for mobile environments using:

* PyTorch Mobile / TorchScript
* Lightweight inference optimizations

---

## 🧰 Built With

* PyTorch
* TorchVision
* NumPy
* Pandas
* Matplotlib

---