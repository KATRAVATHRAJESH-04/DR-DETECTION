"""
voice_engine.py — Voice I/O Engine v2.0
- Text-to-Speech (TTS) via pyttsx3 (offline)
- Speech-to-Text (STT) via SpeechRecognition (Google free API)
- Streamlit-compatible audio player via base64 HTML
"""
import io
import os
import tempfile


# ══════════════════════════════════════════════════════════════════════════════
# TEXT → SPEECH
# ══════════════════════════════════════════════════════════════════════════════

def text_to_speech_bytes(text: str, language: str = "English") -> bytes | None:
    """
    Converts text to speech audio bytes using gTTS (supports multiple languages).

    Returns:
        Audio as bytes (MP3-compatible), or None if gTTS is unavailable.
    """
    if not text or not text.strip():
        return None

    try:
        from gtts import gTTS

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name

        # Map language to gTTS language code, default to English
        lang_map = {
            "English": "en",
            "Hindi": "hi",
            "Spanish": "es",
            "Mandarin": "zh-CN",
            "French": "fr",
            "Arabic": "ar"
        }
        lang_code = lang_map.get(language, "en")

        tts = gTTS(text=text, lang=lang_code)
        tts.save(tmp_path)

        with open(tmp_path, "rb") as f:
            audio_bytes = f.read()

        try:
            os.unlink(tmp_path)
        except OSError:
            pass

        # Return None if the output is suspiciously empty
        if not audio_bytes or len(audio_bytes) < 100:
            return None

        return audio_bytes

    except ImportError:
        print("[Voice Engine] gTTS not installed → pip install gTTS")
        return None
    except Exception as e:
        print(f"[Voice Engine] TTS error: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# SPEECH → TEXT
# ══════════════════════════════════════════════════════════════════════════════

def speech_to_text_from_bytes(audio_bytes: bytes) -> str | None:
    """
    Transcribes a WAV audio clip to text using Google's free STT service
    via the SpeechRecognition library.

    Args:
        audio_bytes: Raw WAV audio bytes.

    Returns:
        Transcribed text string, or None on failure.
    """
    if not audio_bytes:
        return None

    try:
        import speech_recognition as sr

        recognizer  = sr.Recognizer()
        audio_file  = io.BytesIO(audio_bytes)

        # Improve accuracy with adjusted energy threshold
        recognizer.energy_threshold        = 300
        recognizer.dynamic_energy_threshold = True

        with sr.AudioFile(audio_file) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.3)
            audio_data = recognizer.record(source)

        # Try Google first (free, requires internet)
        try:
            text = recognizer.recognize_google(audio_data)
            print(f"[Voice Engine] Transcribed: {text}")
            return text
        except sr.UnknownValueError:
            print("[Voice Engine] Google STT: could not understand audio.")
            return None
        except sr.RequestError as e:
            print(f"[Voice Engine] Google STT unavailable: {e}")
            return None

    except ImportError:
        print("[Voice Engine] SpeechRecognition not installed → pip install SpeechRecognition")
        return None
    except Exception as e:
        print(f"[Voice Engine] STT error: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# STREAMLIT AUDIO PLAYER
# ══════════════════════════════════════════════════════════════════════════════

def create_streamlit_audio_player(audio_bytes: bytes, autoplay: bool = True) -> str:
    """
    Creates an HTML5 audio player for embedding in Streamlit via st.components.v1.html.

    Args:
        audio_bytes: Audio bytes to embed.
        autoplay:    Whether to autoplay the audio (default True).

    Returns:
        HTML string containing the audio player.
    """
    import base64

    if not audio_bytes:
        return ""

    b64 = base64.b64encode(audio_bytes).decode()
    autoplay_attr = "autoplay" if autoplay else ""

    return f"""
    <audio controls {autoplay_attr}
           style="width: 100%; border-radius: 12px; margin-top: 4px;
                  filter: invert(0.85) hue-rotate(180deg) brightness(0.9);">
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        Your browser does not support the audio element.
    </audio>
    """
