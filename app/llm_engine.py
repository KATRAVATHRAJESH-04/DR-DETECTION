"""
app/llm_engine.py — Advanced Gemini AI Explanation Engine v3.0
- Standard explanation (tone-adaptive by severity)
- ELI5 ("Explain Like I'm 10") mode
- Context-aware follow-up Q&A chat
Auto-loads GEMINI_API_KEY from .env in project root.
"""
import os
from pathlib import Path
import google.generativeai as genai

try:
    from google.api_core.exceptions import ResourceExhausted as _QuotaError
except ImportError:
    _QuotaError = None

try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(dotenv_path=_env_path)
except ImportError:
    pass

_ENV_GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")

# ── DR Grade Metadata ──────────────────────────────────────────────────────────
DR_CONTEXT = {
    0: {
        "name":     "No Diabetic Retinopathy",
        "severity": "None",
        "risk":     "Low",
        "tone":     "reassuring and celebratory",
        "meaning":  "Your retina appears healthy with no signs of diabetic damage.",
        "heatmap":  "The model found no significant anomalies; attention is evenly distributed across healthy structures.",
    },
    1: {
        "name":     "Mild Non-Proliferative DR",
        "severity": "Mild",
        "risk":     "Low-Medium",
        "tone":     "calm and encouraging",
        "meaning":  "Small microaneurysms (tiny balloon-like swellings in retinal blood vessels) have been detected.",
        "heatmap":  "Red regions highlight early microaneurysm formations along retinal vasculature.",
    },
    2: {
        "name":     "Moderate Non-Proliferative DR",
        "severity": "Moderate",
        "risk":     "Medium",
        "tone":     "informative and motivating",
        "meaning":  "A larger number of microaneurysms, bleeding spots, or hard exudates are present.",
        "heatmap":  "Red/orange zones indicate multiple lesion sites including exudates and hemorrhage clusters.",
    },
    3: {
        "name":     "Severe Non-Proliferative DR",
        "severity": "Severe",
        "risk":     "High",
        "tone":     "serious but supportive",
        "meaning":  "Many blocked blood vessels are depriving the retina of its blood supply.",
        "heatmap":  "Large red zones indicate widespread vascular blockage and ischemic regions.",
    },
    4: {
        "name":     "Proliferative Diabetic Retinopathy",
        "severity": "Very Severe / Advanced",
        "risk":     "Critical",
        "tone":     "urgent but compassionate",
        "meaning":  "New, fragile blood vessels have grown on the retina (neovascularization), which can bleed and cause vision loss.",
        "heatmap":  "Intense red clusters mark neovascular growth sites and vitreous hemorrhage risk zones.",
    },
}

# ── Model priority list ────────────────────────────────────────────────────────
MODELS_TO_TRY = [
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-2.0-flash-lite",
    "gemini-flash-latest",
]


def _configure_gemini(api_key: str) -> bool:
    """Configures the Gemini client. Returns True if a valid key was found."""
    resolved = (api_key or "").strip() or _ENV_GEMINI_KEY.strip()
    if not resolved:
        return False
    genai.configure(api_key=resolved)
    return True


def _call_gemini(prompt, api_key: str) -> str | None:
    """
    Tries each Gemini model in priority order.
    Returns the response text, or None if all models fail.
    """
    if not _configure_gemini(api_key):
        return None

    last_error = None
    for model_name in MODELS_TO_TRY:
        try:
            model    = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            print(f"[LLM Engine] Used: {model_name}")
            return response.text
        except Exception as e:
            err = str(e)
            is_quota_or_missing = (
                (_QuotaError and isinstance(e, _QuotaError))
                or "429" in err
                or "quota" in err.lower()
                or "ResourceExhausted" in type(e).__name__
                or "404" in err
                or "not found" in err.lower()
                or "403" in err
                or "denied access" in err.lower()
            )
            if is_quota_or_missing:
                print(f"[LLM Engine] {model_name} — quota/unavailable, trying next...")
                last_error = e
                continue
            print(f"[LLM Engine] Error: {e}")
            return None

    print(f"[LLM Engine] All models exhausted. Last: {last_error}")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════════════

def get_gemini_explanation(class_idx: int, confidence: float, api_key: str, language: str = "English") -> str:
    """
    Generates a tone-adaptive, structured medical explanation for a DR result.
    Falls back to a rich template if Gemini is unavailable.
    """
    ctx = DR_CONTEXT.get(class_idx, DR_CONTEXT[0])

    prompt = f"""
You are a warm, intelligent AI medical assistant helping a patient understand their diabetic retinopathy eye scan.
Please write your explanation entirely in this language: {language}.

=== SCAN RESULTS ===
Diagnosis:     {ctx['name']} (Grade {class_idx}/4)
Severity:      {ctx['severity']}
Risk Level:    {ctx['risk']}
AI Confidence: {confidence:.1%}
Detected:      {ctx['meaning']}
Heatmap Notes: {ctx['heatmap']}

=== YOUR TASK ===
Write a clear, compassionate, and professional explanation using a {ctx['tone']} tone.

Structure your response in EXACTLY these 6 sections using markdown:

**📋 Summary**
One sentence summary of what was found.

**🔍 What Was Detected**
Explain what the AI found and why, referencing the heatmap focus regions.

**⚡ Severity & Risk Level**
Describe severity ({ctx['severity']}) and risk ({ctx['risk']}) on a human scale.

**⚠️ Risks If Untreated**
State risks calmly — no scare tactics, but be honest.

**✅ Recommended Actions**
Give 3-5 clear, actionable steps the patient should take.

**💡 Key Takeaway**
One uplifting sentence the patient should remember.

Rules:
- No medical jargon without explanation
- Tone MUST match: {ctx['tone']}
- Total response: ~250 words
"""

    result = _call_gemini(prompt, api_key)
    if result:
        return result
    return _fallback_explanation(class_idx, confidence)


def get_gemini_eli5(class_idx: int, confidence: float, api_key: str, language: str = "English") -> str:
    """
    Generates a child-friendly "Explain Like I'm 10" explanation.
    """
    ctx = DR_CONTEXT.get(class_idx, DR_CONTEXT[0])

    prompt = f"""
You are a kind doctor explaining an eye scan result to a 10-year-old child in very simple, friendly language.
Please write your explanation entirely in this language: {language}.

The eye scan shows: {ctx['name']} (Grade {class_idx}/4)
This means: {ctx['meaning']}
The AI was {confidence:.0%} sure about this.

Explain this as if talking to a curious child using:
- Simple everyday words (no medical terms)
- Fun analogies (like comparing blood vessels to garden hoses)
- Emojis to make it friendly
- A structure of:
  1. What the eye doctor's camera saw 👀
  2. What this means for the person 💭
  3. What happens if we don't help it 😟
  4. What we can do to help 💪
  5. A happy, encouraging ending 🌟

Keep it to about 150 words. Make it friendly, kind, and hopeful.
"""

    result = _call_gemini(prompt, api_key)
    if result:
        return result
    # Simple fallback for ELI5
    simple = {
        0: "👀 Great news! Your eyes look completely healthy! It's like your eye is a beautiful garden with no weeds at all! 🌸 Keep taking your medicine and eating healthy foods, and your eyes will stay happy. See the eye doctor once a year just to check in! 🌟",
        1: "👀 The eye camera found some tiny little bumps on the blood vessels in your eye — like tiny bubbles. 💭 This is very early and small. 💪 If you eat healthy, exercise, and take your medicine, these can get better! See your doctor soon. 🌟",
        2: "👀 The camera found some more bumps and a little bit of leaking in your eye's blood vessels — like a garden hose with small holes. 💭 It needs attention soon. 💪 See your eye doctor within 1-3 months and keep your sugar levels healthy! 🌟",
        3: "👀 Some of the tiny pipes (blood vessels) in your eye are getting blocked, like a clogged garden hose. 😟 This needs to be fixed soon so your eye gets enough food and water. 💪 Please see the eye doctor as soon as you can! 🌟",
        4: "👀 The eye is growing new tiny pipes, but they're fragile and can cause problems — like new, wobbly bridges being built. 😟 This needs help right away from a special eye doctor. 💪 Don't wait — doctors have great tools to help! 🌟",
    }
    return simple.get(class_idx, simple[0])


def get_gemini_chat(
    question: str,
    class_idx: int,
    confidence: float,
    api_key: str,
    history: list,
    language: str = "English"
) -> str:
    """
    Handles context-aware follow-up Q&A about a DR diagnosis.

    Args:
        question:   The user's follow-up question.
        class_idx:  DR grade (0–4).
        confidence: Model confidence (0.0–1.0).
        api_key:    Gemini API key.
        history:    List of {"role": "user"|"ai", "text": "..."} dicts.

    Returns:
        AI response as a string.
    """
    ctx = DR_CONTEXT.get(class_idx, DR_CONTEXT[0])

    # Build conversation history for context
    history_text = ""
    for msg in history[-6:]:  # Last 3 exchanges max
        role = "Patient" if msg["role"] == "user" else "AI Doctor"
        history_text += f"{role}: {msg['text']}\n"

    prompt = f"""
You are an AI-powered ophthalmologist assistant embedded inside a medical web application.
Please strictly write ALL your responses in this language: {language}. Do not switch languages unless explicitly told to.

You behave like a smart, friendly, and professional AI doctor that users can chat with naturally.

----------------------------------------
CONTEXT (VERY IMPORTANT)
----------------------------------------

You ALWAYS have access to the latest patient retinal scan analysis:

- Diabetic Retinopathy Level (0–4): {class_idx} ({ctx['name']})
- Model Confidence: {confidence:.1%}
- Heatmap Insight: {ctx['heatmap']}

You must use this information contextually.

----------------------------------------
YOUR PERSONALITY
----------------------------------------

- Friendly and human-like
- Calm and reassuring
- Professional, like a real doctor
- Conversational and varied (DO NOT respond with the exact same robotic phrasing every time)

----------------------------------------
CORE RESPONSIBILITIES
----------------------------------------

1. Address the patient's question directly.
2. If relevant to the question, explain the result or severity.
   - 0 → No DR
   - 1 → Mild
   - 2 → Moderate
   - 3 → Severe
   - 4 → Proliferative DR
3. Suggest next steps or precautions organically if asked.
4. Keep the user reassured, but factual.

----------------------------------------
HEATMAP EXPLANATION RULE
----------------------------------------

When discussing "why" the AI made its decision, or if asked about the scan/heatmap, you MUST mention the highlighted regions. 
HOWEVER, do NOT use the exact same sentence structure every time. Use natural, varied phrasing. 
Integrate the specific insight naturally: {ctx['heatmap']}
Mention abnormalities like microaneurysms, hemorrhages, or vessel damage.

----------------------------------------
ADAPT TO USER QUESTIONS
----------------------------------------

👉 "Is this serious?"
→ Focus on severity ({ctx['severity']}) + risk level ({ctx['risk']})

👉 "Why did I get this result?" or "Explain heatmap"
→ Explain the heatmap insights and regions focused on.

👉 "What should I do?" or "precautions" or "food habits"
→ Give clear, actionable lifestyle or medical next steps.

----------------------------------------
IMPORTANT RULES & SAFETY
----------------------------------------

- Keep responses CONCISE (3–5 sentences).
- Do NOT sound like a broken record. Vary your wording.
- NEVER invent medical facts.
- Gently remind the patient to consult an eye specialist (but try to paraphrase this so you don't say the exact same "I recommend consulting an eye specialist..." every single time).

----------------------------------------
CONVERSATION HISTORY
----------------------------------------
{history_text if history_text else "This is the first question."}

----------------------------------------
PATIENT'S QUESTION
----------------------------------------
{question}
"""

    result = _call_gemini(prompt, api_key)
    if result:
        return result

    # Fallback chat responses
    return _fallback_chat(question, class_idx, confidence)


def get_gemini_comparison(
    class1: int, conf1: float, img1_b64: str,
    class2: int, conf2: float, img2_b64: str,
    api_key: str, language: str = "English"
) -> str:
    """
    Handles context-aware progression comparison.
    """
    ctx1 = DR_CONTEXT.get(class1, DR_CONTEXT[0])
    ctx2 = DR_CONTEXT.get(class2, DR_CONTEXT[0])

    prompt = f"""
You are an AI ophthalmologist comparing two retinal fundus scans of a patient taken at different times.
Please write your medical summary entirely in this language: {language}.

Scan 1 (Past): {ctx1['name']} (Grade {class1}/4), Confidence: {conf1:.1%}
Scan 2 (Present): {ctx2['name']} (Grade {class2}/4), Confidence: {conf2:.1%}

Analyze the progression between Scan 1 and Scan 2.
State clearly if the condition has Worsened, Improved, or Remained Stable.
Keep it professional, empathetic, and under 200 words. Address the patient directly.
"""

    contents = [prompt]
    
    # Try attaching images to contents
    try:
        from PIL import Image
        from io import BytesIO
        import base64
        if img1_b64:
            contents.append(Image.open(BytesIO(base64.b64decode(img1_b64))).convert("RGB"))
        if img2_b64:
            contents.append(Image.open(BytesIO(base64.b64decode(img2_b64))).convert("RGB"))
    except Exception:
        pass

    result = _call_gemini(contents, api_key)
    if result:
        return result

    if class2 > class1:
        return f"Based on the diagnosis, your condition appears to have **worsened** (Grade {class1} to Grade {class2}). Please consult your ophthalmologist."
    elif class2 < class1:
        return f"Great news—your condition appears to have **improved** (Grade {class1} to Grade {class2}). Continue with your current care plan."
    else:
        return f"Your condition appears to have **remained stable** at Grade {class2}. Continue monitoring as directed by your doctor."


def get_gemini_region_explanation(img_b64: str, x: int, y: int, api_key: str, language: str = "English") -> str:
    """
    Analyzes a specific X, Y pixel coordinate on the image.
    """
    prompt = f"""
You are a highly skilled AI ophthalmologist viewing a retinal fundus image.
The doctor has clicked on a specific pixel coordinate region (X={x}, Y={y}) on this retinal scan. 

Focus strictly on the area at or near this location. 
What medical anomalies (e.g., microaneurysms, hemorrhages, exudates, neovascularization, cotton wool spots) are visible exactly at or near this location?
If the area appears healthy, state that.
Please write your explanation entirely in this language: {language}.
Keep the explanation brief (under 100 words), focused solely on this precise location, and professional.
"""
    contents = [prompt]
    if img_b64:
        try:
            from PIL import Image
            from io import BytesIO
            import base64
            contents.append(Image.open(BytesIO(base64.b64decode(img_b64))).convert("RGB"))
        except Exception:
            pass

    result = _call_gemini(contents, api_key)
    return result if result else "Unable to analyze this specific region right now."


# ══════════════════════════════════════════════════════════════════════════════
# FALLBACK TEMPLATES
# ══════════════════════════════════════════════════════════════════════════════

def _fallback_explanation(class_idx: int, confidence: float) -> str:
    ctx = DR_CONTEXT.get(class_idx, DR_CONTEXT[0])
    templates = {
        0: (
            "**📋 Summary**\nGreat news — no signs of diabetic retinopathy were detected!\n\n"
            "**🔍 What Was Detected**\nYour retinal blood vessels appear healthy and intact. "
            "The AI model found no microaneurysms, hemorrhages, or exudates.\n\n"
            "**⚡ Severity & Risk Level**\nNone — Risk Level: Low. Your eyes are in good shape.\n\n"
            "**⚠️ Risks If Untreated**\nNo current risk, but diabetes can still develop retinopathy over time "
            "if blood sugar is not controlled.\n\n"
            "**✅ Recommended Actions**\n- Continue regular annual eye checkups\n"
            "- Maintain good blood sugar control\n- Follow a healthy, balanced diet\n"
            "- Exercise regularly to support circulation\n\n"
            "**💡 Key Takeaway**\nYour eyes are healthy — keep up the great work!"
        ),
        1: (
            "**📋 Summary**\nEarly signs of mild diabetic retinopathy have been detected.\n\n"
            "**🔍 What Was Detected**\nTiny microaneurysms — small bulges in retinal blood vessels — were identified. "
            "The heatmap shows early vascular changes along peripheral vessels.\n\n"
            "**⚡ Severity & Risk Level**\nMild — Risk Level: Low-Medium. Early stage, manageable with care.\n\n"
            "**⚠️ Risks If Untreated**\nMay progress to moderate or severe DR without intervention.\n\n"
            "**✅ Recommended Actions**\n- Schedule ophthalmologist follow-up within 3-6 months\n"
            "- Tighten blood sugar and blood pressure management\n"
            "- Eat a diet low in refined sugars\n- Exercise at least 150 min/week\n\n"
            "**💡 Key Takeaway**\nCaught early, you have excellent options — act now for best outcomes."
        ),
        2: (
            "**📋 Summary**\nModerate diabetic retinopathy has been detected — medical attention is recommended.\n\n"
            "**🔍 What Was Detected**\nMultiple microaneurysms, retinal hemorrhages, and hard exudates were found. "
            "The heatmap highlights clusters of lesions in multiple retinal zones.\n\n"
            "**⚡ Severity & Risk Level**\nModerate — Risk Level: Medium. Requires active medical management.\n\n"
            "**⚠️ Risks If Untreated**\nProgression to severe DR with potential vision impairment.\n\n"
            "**✅ Recommended Actions**\n- See an ophthalmologist within 1-3 months\n"
            "- Discuss laser photocoagulation therapy\n- Strictly control blood glucose (target HbA1c < 7%)\n"
            "- Monitor blood pressure closely\n\n"
            "**💡 Key Takeaway**\nWith the right medical care, progression can be significantly slowed."
        ),
        3: (
            "**📋 Summary**\nSevere diabetic retinopathy is present — urgent evaluation is needed.\n\n"
            "**🔍 What Was Detected**\nWidespread vascular blockages are depriving significant areas of the retina "
            "of blood supply. The heatmap shows large ischemic zones.\n\n"
            "**⚡ Severity & Risk Level**\nSevere — Risk Level: High. Prompt specialist care required.\n\n"
            "**⚠️ Risks If Untreated**\nHigh probability of advancing to proliferative DR with severe vision loss.\n\n"
            "**✅ Recommended Actions**\n- See a retinal specialist within weeks\n"
            "- Laser photocoagulation therapy likely recommended\n"
            "- Strict diabetes and hypertension control\n- Monthly follow-up monitoring\n\n"
            "**💡 Key Takeaway**\nEarly specialist intervention at this stage can preserve your vision."
        ),
        4: (
            "**📋 Summary**\nProliferative diabetic retinopathy — the most advanced stage — requires immediate care.\n\n"
            "**🔍 What Was Detected**\nAbnormal new blood vessels (neovascularization) have formed on the retina. "
            "The heatmap shows intense activity at neovascular growth sites.\n\n"
            "**⚡ Severity & Risk Level**\nVery Severe — Risk Level: Critical. Immediate specialist care required.\n\n"
            "**⚠️ Risks If Untreated**\nWithout treatment, significant risk of vitreous hemorrhage and blindness.\n\n"
            "**✅ Recommended Actions**\n- **See a retinal specialist immediately**\n"
            "- Anti-VEGF injections or laser surgery options to discuss\n"
            "- Vitrectomy may be considered\n"
            "- Do not delay — early treatment at this stage is vision-saving\n\n"
            "**💡 Key Takeaway**\nModern treatments are highly effective — act immediately to protect your sight."
        ),
    }
    base = templates.get(class_idx, templates[0])
    footer = (
        f"\n\n---\n*🤖 AI Confidence: **{confidence:.1%}** | "
        "This is an AI-generated analysis and does not replace professional medical advice.*"
    )
    return base + footer


def _fallback_chat(question: str, class_idx: int, confidence: float) -> str:
    q = question.lower()
    ctx = DR_CONTEXT.get(class_idx, DR_CONTEXT[0])

    if any(w in q for w in ["serious", "bad", "dangerous", "worried", "scared"]):
        return (
            f"Based on your scan, your diagnosis is **{ctx['name']}** with a risk level of **{ctx['risk']}**. "
            f"{'This is an early finding and very manageable with proper care.' if class_idx <= 1 else 'This does require medical attention — please consult an ophthalmologist soon.'} "
            "Remember, early detection is a significant advantage."
        )
    elif any(w in q for w in ["do", "action", "next", "what should", "recommend"]):
        actions = {
            0: "Your eyes are healthy! Keep up regular annual checkups and maintain good blood sugar control.",
            1: "Schedule a follow-up with an eye doctor within 3-6 months and focus on blood sugar management.",
            2: "See an ophthalmologist within 1-3 months. Discuss treatment options and tighten diabetes control.",
            3: "Please see a retinal specialist as soon as possible — within weeks. Prompt care is essential.",
            4: "**See a retinal specialist immediately.** Anti-VEGF therapy or laser treatment can preserve your vision.",
        }
        return actions.get(class_idx, actions[0])
    elif any(w in q for w in ["heatmap", "red", "color", "highlight", "map"]):
        return (
            f"The Grad-CAM heatmap highlights which retinal regions the AI focused on most. "
            f"For your result: {ctx['heatmap']} "
            "Red = high attention, blue = low attention."
        )
    elif any(w in q for w in ["confidence", "sure", "accurate", "certain"]):
        return (
            f"The AI was **{confidence:.1%}** confident in this diagnosis. "
            "This is based on a hybrid CNN + Vision Transformer model trained on thousands of retinal scans. "
            "Image quality can affect confidence — always confirm with a medical professional."
        )
    else:
        return (
            f"Your diagnosis is **{ctx['name']}** (Grade {class_idx}/4, {ctx['severity']} severity). "
            "For a more detailed answer to your question, I recommend speaking with a qualified ophthalmologist "
            "who can examine your retina in person and provide personalized medical advice."
        )
