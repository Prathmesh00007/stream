
```markdown
# CodeOfDuty Submission - Space Safety Intelligence Console
# We have achieved highest mAP@50=90 and average mAP@50=85

A Streamlit-based application for autonomous perception in high-stakes environments such as space stations.  
This project leverages **YOLO11M** object detection to monitor seven critical safety assets.

---

## 📂 Project Structure

```
.
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
└── AVS325_Submission/
    └── SpaceSafety_Model/
        └── weights/
            └── best.pt             # Custom trained YOLO weights
```

---

## ⚙️ Setup Instructions

Follow these steps to get the project running locally:

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Create and activate a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate      # On Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app
```bash
streamlit run app.py
```

---

## 🧩 Requirements

- Python 3.9+  
- Streamlit  
- Ultralytics YOLO  
- Pillow  
- Pandas  

All dependencies are listed in `requirements.txt`.

---

## 🛰️ Usage

1. Launch the app with `streamlit run app.py`.  
2. Upload a CCTV frame or synthetic render (`.jpg`, `.jpeg`, `.png`).  
3. The app will:
   - Run YOLO11M inference using your trained weights (`best.pt`).  
   - Display detection overlays.  
   - Report confidence scores for seven critical safety assets:
     - OxygenTank  
     - NitrogenTank  
     - FirstAidBox  
     - FireAlarm  
     - SafetySwitchPanel  
     - EmergencyPhone  
     - FireExtinguisher  

---

## ⚠️ Notes

- Ensure your trained weights (`best.pt`) are present in:
  ```
  ./AVS325_Submission/SpaceSafety_Model/weights/best.pt
  ```
- If weights are missing, the app will fall back to default YOLO weights or show an error.  
- For large weight files (>100 MB), consider using [Git LFS](https://git-lfs.github.com/) or hosting them externally (e.g., Hugging Face Hub).

---

## 📡 Demo Objective

This console demonstrates how synthetic data from a Falcon Digital Twin can be used to train robust YOLO detectors for real-world deployment in space safety monitoring.

---
```

---

✨ This README is structured to be **developer-friendly** and **demo-ready**. It covers cloning, installing, running, and troubleshooting.  

Do you want me to also add a **section for deployment on Streamlit Cloud** (with instructions for handling large weight files via Git LFS or Hugging Face Hub)? That would make it easier for others to reproduce your hosted demo.
