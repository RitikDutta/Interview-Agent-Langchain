from __future__ import annotations

import os
from dotenv import load_dotenv

# Load env early so modules can rely on it.
load_dotenv()


# Model names (prefer explicit configuration; keep conservative fallback for dev/test)
GENAI_MODEL = os.getenv("GENAI_MODEL") or os.getenv("GOOGLE_GENAI_MODEL") or "gemini-2.5-flash"
GENAI_MODEL_LOW_TEMP = os.getenv("GENAI_MODEL_LOW_TEMP") or GENAI_MODEL

# Namespaces and indexes
QUESTIONS_NAMESPACE = os.getenv("QUESTIONS_NAMESPACE") or "questions_v4"

