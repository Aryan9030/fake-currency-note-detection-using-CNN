# 💵 Fake Currency Note Detection using CNN

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Flask](https://img.shields.io/badge/Flask-Backend-000000?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#-license)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/Aryan9030)

A web application that detects **fake vs. genuine currency notes** in real time using a **Convolutional Neural Network (CNN)**. Upload a note's image (or use your webcam), and the model instantly classifies it with a confidence score.

> Built by **Aryan** — [github.com/Aryan9030](https://github.com/Aryan9030)

---
live server : https://verinote-cnn-currency-ok5v.arcada.app
## 📖 Table of Contents

- [About the Project](#-about-the-project)
- [Features](#-features)
- [Demo](#-demo)
- [Tech Stack](#-tech-stack)
- [How It Works](#-how-it-works)
- [CNN Architecture](#-cnn-architecture)
- [Dataset](#-dataset)
- [Results](#-results)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [API Endpoints](#-api-endpoints)
- [Training the Model](#-training-the-model)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 🎯 About the Project

Counterfeit currency is a growing concern worldwide, causing financial losses to individuals, businesses, and governments. Manual verification is slow and error-prone.

This project automates the detection process using **Deep Learning**. A CNN model is trained on images of genuine and counterfeit notes, learning visual differences such as:

- 🔍 Print quality and sharpness
- 🎨 Color consistency and ink patterns
- 🧵 Security thread and watermark regions
- 🔢 Serial number typography
- 📐 Texture and micro-printing details

The trained model is served through a **Flask web app**, making it accessible from any browser.

---

## ✨ Features

- 🖼️ **Image Upload Detection** — upload a photo of a currency note and get an instant verdict
- 📷 **Webcam / Live Detection** *(optional module)* — classify frames in real time
- 🧠 **CNN-based Classification** — custom Convolutional Neural Network built with TensorFlow/Keras
- 📊 **Confidence Score** — every prediction includes the model's confidence (%)
- ⚡ **Fast Inference** — predictions in under a second on CPU
- 🌐 **Simple Web UI** — clean, responsive HTML/CSS/JavaScript frontend
- 🔌 **REST API** — `/predict` endpoint for programmatic access
- 📈 **Training Pipeline** — complete scripts for data augmentation, training, and evaluation

---

## 🎬 Demo

| Genuine Note ✅ | Fake Note ❌ |
|:---:|:---:|
| ![Genuine prediction](static/demo/genuine.gif) | ![Fake prediction](static/demo/fake.gif) |

> *Add your demo GIFs/screenshots to `static/demo/` and they will render here.*

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| Deep Learning | TensorFlow / Keras (CNN) |
| Backend | Python, Flask |
| Image Processing | OpenCV, NumPy, Pillow |
| Frontend | HTML5, CSS3, JavaScript (Fetch API) |
| Model Serialization | Keras `.h5` / SavedModel format |
| Deployment | Gunicorn (optional), Render / Railway / AWS ready |

---

## ⚙️ How It Works

```
┌──────────────┐   ┌──────────────────┐   ┌──────────────────┐   ┌────────────────┐
│  User uploads │   │  Preprocessing   │   │   CNN Model      │   │   Result       │
│  note image   │──▶│  resize 224×224  │──▶│  (trained weights│──▶│  FAKE/GENUINE  │
│  via web UI   │   │  normalize pixels│   │   inference)     │   │  + confidence  │
└──────────────┘   └──────────────────┘   └──────────────────┘   └────────────────┘
```

1. **Upload** — user submits a currency note image through the web interface.
2. **Preprocess** — the image is resized to `224 × 224`, normalized, and batched.
3. **Predict** — the CNN runs inference and outputs class probabilities.
4. **Respond** — the Flask backend returns the label (`FAKE` / `GENUINE`) and confidence to the UI.

---

## 🧬 CNN Architecture

A custom sequential CNN (4 convolutional blocks + dense head):

| Layer | Output Shape | Params |
|---|---|---|
| Conv2D 32 + ReLU + MaxPool | (224, 224, 32) → pooled | ~900 |
| Conv2D 64 + ReLU + MaxPool | ↓ | ~18K |
| Conv2D 128 + ReLU + MaxPool | ↓ | ~74K |
| Conv2D 128 + ReLU + MaxPool | ↓ | ~147K |
| Flatten + Dropout(0.5) | — | 0 |
| Dense 512 + ReLU | (512,) | ~large |
| Dense 1 + Sigmoid | (1,) | binary output |

- **Optimizer:** Adam (`lr = 1e-4`)
- **Loss:** Binary Cross-Entropy
- **Augmentation:** rotation, flip, zoom, brightness shifts to improve robustness

---

## 📚 Dataset

The model is trained on a binary dataset of currency note images:

- ✅ **Genuine notes** — high-resolution scans of authentic notes
- ❌ **Fake notes** — images of known counterfeit samples
- 🔄 **Augmented** — on-the-fly augmentation (rotation ±20°, horizontal/vertical flips, zoom, brightness jitter) to expand the training set and reduce overfitting

Data is split **80% train / 10% validation / 10% test**, and all images are resized to `224 × 224 × 3`.

---

## 📈 Results

| Metric | Score |
|---|---|
| Training Accuracy | ~98% |
| Validation Accuracy | ~96% |
| Test Accuracy | ~95% |
| Precision (Fake) | 0.94 |
| Recall (Fake) | 0.93 |
| F1-Score | 0.93 |
| Inference Time (CPU) | < 1 s / image |

> Update these numbers with your actual training run results from `history.json`.

---

## 📂 Project Structure

```
fake-currency-detector/
├── app.py                  # Flask application entry point
├── model/
│   ├── train.py            # CNN training script
│   ├── predict.py          # standalone inference script
│   └── currency_cnn.h5     # trained model weights (download separately)
├── static/
│   ├── css/style.css       # UI styling
│   ├── js/script.js        # upload + fetch logic
│   ├── demo/               # demo GIFs / screenshots
│   └── uploads/            # temporary uploaded images
├── templates/
│   └── index.html          # main web page
├── dataset/                # train/val/test image folders (not committed)
│   ├── genuine/
│   └── fake/
├── requirements.txt        # Python dependencies
├── LICENSE
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites

- Python **3.9+**
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Aryan9030/fake-currency-detector.git
cd fake-currency-detector

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place your trained model weights at model/currency_cnn.h5

# 5. Run the web app
python app.py
```

Open **http://localhost:5000** in your browser. 🎉

### `requirements.txt`

```txt
tensorflow>=2.10
flask>=2.2
opencv-python>=4.6
numpy>=1.23
pillow>=9.0
matplotlib>=3.6
```

---

## 🕹 Usage

1. Launch the app (`python app.py`).
2. Go to `http://localhost:5000`.
3. Click **"Choose Image"** and select a currency note photo (JPG/PNG).
4. Click **"Detect"**.
5. The page displays **GENUINE ✅** or **FAKE ❌** along with the confidence percentage.

**CLI inference** (without the web UI):

```bash
python model/predict.py --image path/to/note.jpg
# Output: PREDICTION: FAKE | Confidence: 97.32%
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the web UI |
| `POST` | `/predict` | Accepts a multipart image upload, returns JSON |
| `GET` | `/health` | Health check |

**Example request:**

```bash
curl -X POST -F "image=@note.jpg" http://localhost:5000/predict
```

**Example response:**

```json
{
  "prediction": "FAKE",
  "confidence": 0.9732,
  "time_ms": 412
}
```

---

## 🏋️ Training the Model

To train the CNN from scratch on your own dataset:

```bash
# Organize images as dataset/genuine/*.jpg and dataset/fake/*.jpg
python model/train.py --data dataset/ --epochs 25 --batch-size 32
```

Training artifacts are saved automatically:
- `model/currency_cnn.h5` — model weights
- `history.json` — accuracy/loss curves data
- `confusion_matrix.png` — evaluation plot

---

## 🗺 Roadmap

- [ ] Multi-currency support (USD, EUR, INR, ...)
- [ ] Mobile-friendly PWA version
- [ ] Grad-CAM heatmaps to visualize *why* a note was flagged
- [ ] Transfer learning variant (MobileNetV2 / EfficientNet) for higher accuracy
- [ ] Docker image for one-command deployment
- [ ] Android app wrapper

---

## 🤝 Contributing

Contributions make this project better — they're **greatly appreciated**!

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more information.

---

## 📬 Contact

**Aryan**

- GitHub: [@Aryan9030](https://github.com/Aryan9030)
- Project Link: [github.com/Aryan9030/fake-currency-detector](https://github.com/Aryan9030/fake-currency-detector)

---

<p align="center">
  ⭐ If you found this project helpful, please give it a star! ⭐
</p>
