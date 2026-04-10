# 👁️ DR AI Healthcare Assistant

A **production-ready, full-stack AI system** for early detection of Diabetic Retinopathy using retinal fundus images.

Built with a **Hybrid CNN + Vision Transformer** architecture, **Grad-CAM** explainability, **Gemini AI** natural-language explanations, and **voice I/O**.

---

## 🧠 Full Pipeline

```
User → Streamlit UI → FastAPI → CNN+ViT Model → Grad-CAM → Gemini LLM → Voice Output → UI
```

---

## 🗂️ Project Structure

```
project/
├── api/
│   └── main.py           # FastAPI backend (/predict, /explain, /health)
├── app/
│   ├── ui.py             # Streamlit frontend (premium dark UI)
│   ├── database.py       # SQLite user & prediction history storage
│   ├── llm_engine.py     # Gemini AI explanation engine (with offline fallback)
│   └── voice_engine.py   # pyttsx3 TTS + SpeechRecognition STT
├── models/
│   └── hybrid_model.py   # Hybrid CNN + ViT architecture
├── utils/
│   ├── gradcam.py        # Grad-CAM heatmap generation
│   └── preprocessing.py  # Retinal image preprocessing
├── inference.py          # End-to-end inference class
├── training/train.py     # Model training script (Kaggle/GPU)
├── dr_hybrid_model.pth   # Trained model weights
└── requirements.txt
```

---

## ✨ Features

| Feature | Details |
|---|---|
| 🔬 **DR Detection** | 5-class classification (Grade 0–4) |
| 🔥 **Grad-CAM** | Heatmap overlay showing affected retinal regions |
| 🧠 **AI Explanation** | Gemini API or offline template-based explanation |
| 🎤 **Voice Input** | WAV upload → SpeechRecognition STT |
| 🔊 **Voice Output** | pyttsx3 offline TTS, plays in-browser |
| 🔐 **Auth** | Persistent SQLite login/signup |
| 📋 **Reports** | CSV + Plain-text downloadable prediction history |
| 🎨 **Premium UI** | Dark theme, Grad-CAM side-by-side, confidence meters |

---

## 🚀 Running From Scratch

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start the FastAPI Backend (Terminal 1)

```bash
uvicorn api.main:app --reload --port 8000
```

### 3. Start the Streamlit UI (Terminal 2)

```bash
streamlit run app/ui.py
```

> ℹ️ Open http://localhost:8501 in your browser.  
> Default login: **admin / admin**

---

## 🔑 Gemini API Key (Optional)

To enable AI-powered explanations:
1. Get a free key at https://aistudio.google.com/app/apikey
2. Paste it in the **⚙️ AI Configuration** section in the sidebar

> Without a key, the system uses built-in medical templates automatically.

---

## 🩺 DR Grading Scale

| Grade | Diagnosis | Action |
|---|---|---|
| 0 | No DR | Annual checkup |
| 1 | Mild NPDR | Monitor, lifestyle changes |
| 2 | Moderate NPDR | Ophthalmologist in 1–3 months |
| 3 | Severe NPDR | Urgent evaluation |
| 4 | Proliferative DR | **Immediate specialist care** |

---

> ⚠️ **Disclaimer**: This system is for research and educational purposes. It does not replace professional medical diagnosis.
