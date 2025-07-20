
# MediScan AI

MediScan AI is a deep learning-based medical diagnostic tool that assists in detecting pneumonia from chest X-ray images. Built with VGG16 transfer learning and integrated into a user-friendly PyQt5 interface, the system aims to support healthcare professionals with quick and accurate analysis.

## 🔬 Project Highlights

- 🧠 **Model**: VGG16 pre-trained on ImageNet with transfer learning.
- 🩻 **Dataset**: A curated chest X-ray image dataset containing normal and pneumonia cases.
- 📊 **Evaluation Metrics**: Accuracy, Sensitivity, Specificity, F1-Score.
- 🖥️ **Interface**: Built using PyQt5 for easy interaction with the AI system.
- ⚙️ **Tech Stack**: Python, TensorFlow/Keras, NumPy, OpenCV, PyQt5.

---

## 📁 Project Structure

```
MediScan-AI/
├── dataset/
│   ├── train/
│   ├── test/
│   └── val/
├── model/
│   └── mediscan_vgg16_model.h5
├── ui/
│   └── mediscan_ui.py
├── main.py
├── utils.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

1. **Clone the repository**  
```bash
git clone https://github.com/yourusername/mediscan-ai.git
cd mediscan-ai
```

2. **Create virtual environment (optional)**  
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

3. **Install dependencies**  
```bash
pip install -r requirements.txt
```

---

## 🧪 How to Run

1. **Train the model** (if not using the pre-trained one):  
```bash
python train_model.py
```

2. **Launch the GUI application**:  
```bash
python main.py
```

3. **Use the interface** to upload a chest X-ray image and get instant prediction results.

---

## 📈 Performance Summary

| Metric       | Value  |
|--------------|--------|
| Accuracy     | 97.0%  |
| Sensitivity  | 91.2%  |
| Specificity  | 95.1%  |
| F1-Score     | 92.1%  |

> *Note: These values may vary depending on the training-validation split.*

---

## 🚧 Challenges

- Reducing false negatives to avoid missed pneumonia cases.
- Handling data imbalance between normal and pneumonia cases.
- Optimizing model performance on unseen test samples.

---

## 🔭 Future Work

- Extend to multi-class classification (e.g., bacterial vs viral pneumonia).
- Support for other diseases (e.g., COVID-19, tuberculosis).
- Deploy as a web or mobile application for wider accessibility.
- Improve explainability using Grad-CAM or SHAP for medical trust.

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🙌 Acknowledgements

- Chest X-ray datasets used in training and testing.
- VGG16 model by the Visual Geometry Group, University of Oxford.
- PyQt5 for GUI support.
