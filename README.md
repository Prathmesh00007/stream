## 📝 Description

CodeOfDuty Submission. We were tasked to create a robust system which can detect 7 space safety equipments in low-light conditions. We have fine-tuned and hypertuned YOLOv 11M model to achieve highest mAP@50 =  90 and average mAP@50 = 85

## 🛠️ Tech Stack

- Python
- Ultralytics
- PyTorch
- Stramlit
- YOLOv 11M
- OpenCV
- Falcon Digital Twin Platform


## 📦 Key Dependencies

```
streamlit: 1.51.0
ultralytics: 8.3.232
torch: 2.0.0
torchvision: 0.15.0
pandas: latest
numpy: latest
opencv-python-headless: latest
Pillow: latest
```

## 📁 Project Structure

```
.
├── .devcontainer
│   └── devcontainer.json
├── packages.txt
└── prodReadyStreamlit
    ├── AVS325_Submission
    │   ├── SpaceSafety_Model
    │   │   ├── args.yaml
    │   │   └── labels.jpg
    │   ├── SpaceSafety_Model2
    │   │   ├── args.yaml
    │   │   ├── labels.jpg
    │   │   ├── train_batch0.jpg
    │   │   └── train_batch1.jpg
    │   └── SpaceSafety_Model3
    │       ├── BoxF1_curve.png
    │       ├── BoxPR_curve.png
    │       ├── BoxP_curve.png
    │       ├── BoxR_curve.png
    │       ├── args.yaml
    │       ├── confusion_matrix.png
    │       ├── confusion_matrix_normalized.png
    │       ├── labels.jpg
    │       ├── results.csv
    │       ├── results.png
    │       ├── train_batch0.jpg
    │       ├── train_batch1.jpg
    │       ├── train_batch2.jpg
    │       ├── train_batch37570.jpg
    │       ├── train_batch37571.jpg
    │       ├── train_batch37572.jpg
    │       ├── val_batch0_labels.jpg
    │       ├── val_batch0_pred.jpg
    │       ├── val_batch1_labels.jpg
    │       ├── val_batch1_pred.jpg
    │       ├── val_batch2_labels.jpg
    │       ├── val_batch2_pred.jpg
    │       └── weights
    │           ├── best.pt
    │           └── last.pt
    ├── app.py
    ├── requirements.txt
    ├── train_model.py
    └── yolo11m.pt
```

## 🛠️ Development Setup

### Python Setup
1. Install Python (v3.8+ recommended)
2. Create a virtual environment: `python -m venv venv`
3. Activate the environment:
   - Windows: `venv\Scripts\activate`
   - Unix/MacOS: `source venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Run: streamlit run app.py

## 👥 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork** the repository
2. **Clone** your fork: `git clone https://github.com/Prathmesh00007/stream.git`
3. **Create** a new branch: `git checkout -b feature/your-feature`
4. **Commit** your changes: `git commit -am 'Add some feature'`
5. **Push** to your branch: `git push origin feature/your-feature`
6. **Open** a pull request

Please ensure your code follows the project's style guidelines and includes tests where applicable.

---
*This README was generated with ❤️ by ReadmeBuddy*
