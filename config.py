# config.py
import os

# ------------------------------
# Helper(s)
# ------------------------------
def _env(name: str, default: str = "") -> str:
    """Get environment variable with a safe default."""
    val = os.getenv(name, default)
    return val.strip() if isinstance(val, str) else val

def _env_int(name: str, default: int) -> int:
    """Get environment variable as int with fallback."""
    try:
        return int(os.getenv(name, default))
    except Exception:
        return default


# ------------------------------
# Database
# ------------------------------
DATABASE_URL = _env("DATABASE_URL", "sqlite:///./synermind.db")


# ------------------------------
# App Security
# ------------------------------
SECRET_KEY = _env("SECRET_KEY", "dev-secret")


# ------------------------------
# LLM — Gemini (for high-level reasoning & routing)
# ------------------------------
# For convenience we also accept GOOGLE_API_KEY if it’s set.
GEMINI_API_KEY = _env("GEMINI_API_KEY", _env("GOOGLE_API_KEY", "AIzaSyC69iMg4_lkPbx2kAEgGfx8YWUcKAEgo1Q"))


# ------------------------------
# LLM — Groq (for fast agent conversation)
# ------------------------------
# Runs open-source models on ultra-fast LPUs for responsive chat.
GROQ_API_KEY = _env("GROQ_API_KEY", "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")


# ------------------------------
# Email / SMTP (for alerts & user verification)
# ------------------------------
EMAIL_USER  = _env("EMAIL_USER", "")
EMAIL_PASS  = _env("EMAIL_PASS", "")

# --- SendGrid API (for alerts & user verification) ---
SENDGRID_API_KEY = _env("SENDGRID_API_KEY", "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")
SENDGRID_FROM_EMAIL = _env("SENDGRID_FROM_EMAIL", "priya0401rai@gmail.com")
#SMTP_SERVER = _env("SMTP_SERVER", "smtp.gmail.com")
#SMTP_PORT   = _env_int("SMTP_PORT", 587)


# ------------------------------
# Front-end URLs for deep links
# ------------------------------
FRONTEND_BASE_URL = _env("FRONTEND_BASE_URL", "http://localhost:8501").rstrip("/")
VERIFY_LINK_TPL = f"{FRONTEND_BASE_URL}?verify={{token}}"
RESET_LINK_TPL  = f"{FRONTEND_BASE_URL}?reset={{token}}"