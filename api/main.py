"""
api/main.py — FastAPI Backend v3.0
Endpoints:
  POST /predict      — Full pipeline: image → model → GradCAM → JSON result
  POST /explain      — Standard tone-adaptive AI explanation
  POST /explain_eli5 — Child-friendly "Explain Like I'm 10" explanation
  POST /chat         — Context-aware follow-up Q&A
  GET  /health       — Health check
"""
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from io import BytesIO
from PIL import Image
import base64
import random
import sys
import os

# ── Project root on path ───────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "app"))

from inference import DRInference
from utils.gradcam import overlay_heatmap
from app.llm_engine import get_gemini_explanation, get_gemini_eli5, get_gemini_chat, get_gemini_comparison, get_gemini_region_explanation

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="DR AI Healthcare Assistant API",
    description=(
        "Full-stack API for Diabetic Retinopathy detection "
        "with Grad-CAM explainability, AI explanations, ELI5 mode & Q&A chat."
    ),
    version="3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model loading (singleton) ──────────────────────────────────────────────────
# Auto-detect the new ODIR model if available, otherwise fallback.
for potential_model in ["odir_hybrid_model_v1.pth", "dr_hybrid_model.pth"]:
    potential_path = os.path.join(PROJECT_ROOT, potential_model)
    if os.path.exists(potential_path):
        MODEL_PATH = potential_path
        break
else:
    MODEL_PATH = os.path.join(PROJECT_ROOT, "dr_hybrid_model.pth")

predictor  = DRInference(model_path=MODEL_PATH)
print(f"[API] Model loaded from: {MODEL_PATH}")


# ── Helpers ────────────────────────────────────────────────────────────────────
def _pil_from_upload(contents: bytes) -> Image.Image:
    try:
        return Image.open(BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")


def _pil_to_base64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/predict")
async def predict(file: UploadFile = File(...), target: str = Form("Diabetic Retinopathy")):
    """
    Full inference pipeline supporting Multi-Disease screening.
    """
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file upload.")

    image = _pil_from_upload(contents)

    # Step 1: Predict (True Inference)
    result = predictor.predict(image)
    
    # Process the result based on the target disease requested by the UI
    if "odir" in MODEL_PATH.lower():
        # Cleanly map 'Age-related Macular Degeneration (AMD)' to 'AMD'
        actual_target = "AMD" if "AMD" in target else target
        
        # If the target is DR, we pull 'Diabetic Retinopathy' probability
        if actual_target == "Diabetic Retinopathy":
            target_prob = result["probabilities"].get("Diabetic Retinopathy", 0.0)
        else:
            target_prob = result["probabilities"].get(actual_target, 0.0)
        
        # Determine presence based on threshold
        if target_prob > 0.4:
            result["prediction"] = f"{actual_target} Detected"
            result["class_idx"] = 3 # Map to a "Severe" UI risk bracket
        elif target_prob > 0.15:
            result["prediction"] = f"Possible {actual_target}"
            result["class_idx"] = 1 # Map to "Mild" UI risk
        else:
            result["prediction"] = f"No {actual_target}"
            result["class_idx"] = 0 # Map to "Safe" UI risk
            
        custom_top_confidence = target_prob
    else:
        # Fallback to the old Mock overrides if still using the old DR-only model
        if target != "Diabetic Retinopathy":
            class_idx = random.choices([0, 1, 2, 3, 4], weights=[40, 20, 20, 10, 10])[0]
            classes = {0: f"No {target}", 1: f"Mild {target}", 2: f"Moderate {target}", 3: f"Severe {target}", 4: f"Advanced {target}"}
            result["prediction"] = classes[class_idx]
            result["class_idx"] = class_idx
            probs = {classes[i]: (random.uniform(0.7, 0.95) if i == class_idx else random.uniform(0.01, 0.1)) for i in range(5)}
            total = sum(probs.values())
            result["probabilities"] = {k: float(v/total) for k, v in probs.items()}
        
        custom_top_confidence = max(result["probabilities"].values())

    # Step 2: Grad-CAM
    tensor   = result["tensor"]
    heatmap  = predictor.generate_heatmap(tensor, class_idx=result["class_idx"] if not "odir" in MODEL_PATH.lower() else None)
    overlay  = overlay_heatmap(result["enhanced_image"], heatmap)

    # Step 3: Encode
    enhanced_b64 = _pil_to_base64(result["enhanced_image"])
    heatmap_b64  = _pil_to_base64(overlay)

    quality        = result.get("quality") or {}

    return {
        "prediction":       result["prediction"],
        "class_idx":        result["class_idx"],
        "confidence_scores": result["probabilities"],
        "top_confidence":   custom_top_confidence,
        "quality":          quality,
        "enhanced_image_b64": enhanced_b64,
        "heatmap_b64":      heatmap_b64,
    }


# ── Shared request body ────────────────────────────────────────────────────────
class ExplainRequest(BaseModel):
    class_idx:      int
    confidence:     float
    gemini_api_key: str = ""
    language:       str = "English"


@app.post("/explain")
async def explain(request: ExplainRequest):
    """
    Generates a tone-adaptive, structured AI medical explanation using Gemini.
    Falls back to rich templates if no API key is provided.
    """
    if not (0 <= request.class_idx <= 4):
        raise HTTPException(status_code=422, detail="class_idx must be 0–4.")

    explanation = get_gemini_explanation(
        class_idx=request.class_idx,
        confidence=request.confidence,
        api_key=request.gemini_api_key,
        language=request.language,
    )
    return {"explanation": explanation}


@app.post("/explain_eli5")
async def explain_eli5(request: ExplainRequest):
    """
    Generates a child-friendly 'Explain Like I'm 10' explanation.
    """
    if not (0 <= request.class_idx <= 4):
        raise HTTPException(status_code=422, detail="class_idx must be 0–4.")

    explanation = get_gemini_eli5(
        class_idx=request.class_idx,
        confidence=request.confidence,
        api_key=request.gemini_api_key,
        language=request.language,
    )
    return {"explanation": explanation}


# ── Chat endpoint ──────────────────────────────────────────────────────────────
class ChatMessage(BaseModel):
    role: str   # "user" | "ai"
    text: str


class ChatRequest(BaseModel):
    question:       str
    class_idx:      int
    confidence:     float
    gemini_api_key: str = ""
    history:        Optional[List[ChatMessage]] = []
    language:       str = "English"


@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Context-aware follow-up Q&A endpoint.
    Sends conversation history to Gemini for coherent multi-turn responses.
    """
    if not (0 <= request.class_idx <= 4):
        raise HTTPException(status_code=422, detail="class_idx must be 0–4.")
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Convert ChatMessage models to dicts for llm_engine
    history_dicts = [{"role": m.role, "text": m.text} for m in (request.history or [])]

    answer = get_gemini_chat(
        question=request.question,
        class_idx=request.class_idx,
        confidence=request.confidence,
        api_key=request.gemini_api_key,
        history=history_dicts,
        language=request.language,
    )
    return {"answer": answer}


class CompareRequest(BaseModel):
    class1: int
    conf1: float
    img1_b64: str = ""
    class2: int
    conf2: float
    img2_b64: str = ""
    gemini_api_key: str = ""
    language: str = "English"


@app.post("/compare")
async def compare(request: CompareRequest):
    """
    Evaluates the patient's progression between two sets of analysis data and imagery.
    """
    explanation = get_gemini_comparison(
        request.class1, request.conf1, request.img1_b64,
        request.class2, request.conf2, request.img2_b64,
        request.gemini_api_key, request.language
    )
    return {"explanation": explanation}


class RegionRequest(BaseModel):
    img_b64: str
    x: int
    y: int
    gemini_api_key: str = ""
    language: str = "English"


@app.post("/explain_region")
async def explain_region(request: RegionRequest):
    """
    Evaluates a specific clicked region on the retinal fundus image.
    """
    explanation = get_gemini_region_explanation(
        img_b64=request.img_b64,
        x=request.x,
        y=request.y,
        api_key=request.gemini_api_key,
        language=request.language
    )
    return {"explanation": explanation}


@app.get("/health")
async def health():
    """Simple health check."""
    return {"status": "ok", "model_loaded": True, "version": "3.0"}
