# gemini_core.py
from pathlib import Path
import os, json, requests, textwrap
from dotenv import load_dotenv # Added this for .env support

# Load environment variables from .env file
load_dotenv()

# 1) Try to get your key from an environment variable (preferable)
KEY = os.getenv("GeminiKey")

if not KEY:
    # Fallback to a config file if environment variable is not set (less common for Windows)
    cfg = Path.home() / ".config/gemini_api_key"
    KEY = cfg.read_text().strip() if cfg.exists() else ""

if not KEY:                                     # fail fast so the caller sees it
    raise RuntimeError("❌  GEMINI_API_KEY missing in env or ~/.config/gemini_api_key")

_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=" + KEY

def ask_gemini(prompt:str, *, model:str="gemini-1.5-flash", max_tokens:int=256)->str:
    """Light-weight wrapper – no external deps, <15 KB/s traffic."""
    body = {
        "model": model,
        "contents":[{"role":"user","parts":[{"text": prompt}]}],
        "generation_config":{"max_output_tokens":max_tokens,"temperature":0.6}
    }
    try:
        r = requests.post(_ENDPOINT, json=body, timeout=30)
        r.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        response_json = r.json()
        if "candidates" in response_json and response_json["candidates"]:
            return response_json["candidates"][0]["content"]["parts"][0]["text"].strip()
        else:
            return f"⚠️ Gemini error: No candidates in response. Raw: {response_json}"
    except requests.exceptions.Timeout:
        return "⚠️ Gemini error: Request timed out."
    except requests.exceptions.RequestException as e:
        return f"⚠️ Gemini error: Request failed: {e}"
    except json.JSONDecodeError:
        return f"⚠️ Gemini error: Could not decode JSON response. Raw: {r.text}"
    except Exception as e:
        return f"⚠️  Gemini unexpected error: {e}"