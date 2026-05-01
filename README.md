# 👁️ Ocular Disease AI Healthcare Assistant

A **production-ready, full-stack AI system** for early detection of multiple ocular diseases including Diabetic Retinopathy (DR), Glaucoma, Age-related Macular Degeneration (AMD), Cataracts, and more, using retinal fundus images.

Built with a **Hybrid CNN + Vision Transformer** architecture trained on the ODIR-5K dataset, featuring **Grad-CAM** explainability, **Gemini AI** natural-language explanations, progression tracking, and an interactive chat.

---

## 🧠 Full Pipeline

```text
User/Doctor → Streamlit UI → FastAPI → CNN+ViT Model → Grad-CAM → Gemini LLM → Voice Output → UI
```

---

## 🗂️ Project Structure

```text
project/
├── api/
│   └── main.py           # FastAPI backend (/predict, /explain, /compare, /chat)
├── app/
│   ├── ui.py             # Streamlit frontend (premium dark UI, patient/doctor portals)
│   ├── database.py       # SQLite user & prediction history storage
│   ├── llm_engine.py     # Gemini AI explanation engine (with offline fallback)
│   └── voice_engine.py   # pyttsx3 TTS + SpeechRecognition STT
├── models/
│   └── hybrid_model.py   # Hybrid CNN + ViT architecture (supports 8-class & 5-class)
├── utils/
│   ├── gradcam.py        # Grad-CAM heatmap generation
│   └── preprocessing.py  # Retinal image preprocessing
├── inference.py          # End-to-end inference class
├── training/train.py     # Model training script
├── odir_hybrid_model_v1.pth # Trained ODIR multi-disease model weights
├── dr_hybrid_model.pth   # Legacy DR-only model weights
└── requirements.txt
```

---

## ✨ Features

| Feature | Details |
|---|---|
| 🔬 **Multi-Disease Detection** | 8-class detection (Normal, DR, Glaucoma, Cataract, AMD, Hypertension, Myopia, Other) |
| 🔥 **Grad-CAM** | Interactive heatmap overlay showing affected retinal regions (click regions for detailed explanation) |
| 🧠 **AI Explanation** | Gemini API medical explanations with child-friendly "ELI5" mode |
| 📈 **Progression Tracking** | Compare past and present scans with AI-generated progression summaries |
| 🤖 **AI Chat** | Ask follow-up questions to the AI Doctor based on your scan results |
| 🔊 **Voice I/O** | pyttsx3 offline TTS and SpeechRecognition STT capabilities |
| 🔐 **Role-Based Portals** | Dedicated interfaces for Patients and Doctors with persistent SQLite storage |
| 📋 **Reports** | Downloadable PDF and CSV prediction history reports |
| 🎨 **Premium UI** | Dark theme, glassmorphism, animated UI, and confidence meters |

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
> You can sign up as a Patient or Doctor from the login screen.

---

## 🔑 Gemini API Key (Optional)

To enable AI-powered explanations, Q&A chat, and progression summaries:
1. Get a free key at https://aistudio.google.com/app/apikey
2. Add it to a `.env` file in the root directory: `GEMINI_API_KEY=your_key_here`

> Without a key, the system uses built-in medical templates automatically.

---

## 🩺 Supported Ocular Diseases

| Disease | Description | Action |
|---|---|---|
| **Normal** | Healthy retina | Annual checkup |
| **Diabetic Retinopathy** | Damage to retinal blood vessels | Monitor, lifestyle changes, or specialist care depending on severity |
| **Glaucoma** | Optic nerve damage, often linked to eye pressure | Ophthalmologist evaluation |
| **Cataract** | Clouding of the normally clear lens of the eye | Surgical consultation if vision impaired |
| **AMD** | Damage to the macula affecting central vision | Urgent specialist evaluation |

---

> ⚠️ **Disclaimer**: This system is for research and educational purposes. It does not replace professional medical diagnosis.
