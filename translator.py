from deep_translator import GoogleTranslator
from gtts import gTTS
import tempfile

# ✅ Single language mapping for BOTH translation and speech
LANGUAGE_CODES = {
    "English": "en",
    "Hindi": "hi",
    "Kannada": "kn",
    "Tamil": "ta",
    "Telugu": "te",
    "Spanish": "es",
    "French": "fr",
    "German": "de"
}


def translate_batch(texts, language, device=None):
    """
    Translate a list of captions into selected language.
    """

    # If English selected → no translation needed
    if language == "English":
        return texts

    target_lang = LANGUAGE_CODES.get(language)

    # If unsupported language → return original text
    if not target_lang:
        return texts

    translator = GoogleTranslator(source="en", target=target_lang)

    translated = []
    for text in texts:
        try:
            translated_text = translator.translate(text)
            translated.append(translated_text)
        except Exception as e:
            print(f"Translation failed: {e}")
            translated.append(text)

    return translated


def text_to_speech(text, language):
    """
    Convert text into speech using same language mapping.
    """

    lang_code = LANGUAGE_CODES.get(language, "en")

    try:
        tts = gTTS(text=text, lang=lang_code)

        # Create temporary audio file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_file.name)

        return temp_file.name

    except Exception as e:
        print("TTS error:", e)
        return None