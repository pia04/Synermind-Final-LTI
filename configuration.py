# config.py
import streamlit as st

# This is the new, secure way to access your secret keys.
# It works both locally (reading from .streamlit/secrets.toml)
# and on Streamlit Cloud (reading from the secrets you paste in the dashboard).

# --- Helper Function ---
def get_secret(key, default=None):
    """A helper function to safely get a secret value."""
    if hasattr(st, 'secrets') and key in st.secrets:
        return st.secrets[key]
    return default

# ------------------------------
# Database
# ------------------------------
DATABASE_URL = get_secret("DATABASE_URL", "sqlite:///./synermind.db")

# ------------------------------
# App Security
# ------------------------------
SECRET_KEY = get_secret("SECRET_KEY", "dev-secret")

# ------------------------------
# LLM API Keys
# ------------------------------
GEMINI_API_KEY = get_secret("GEMINI_API_KEY")
GROQ_API_KEY = get_secret("GROQ_API_KEY")

# ------------------------------
# SendGrid API (for email)
# ------------------------------
SENDGRID_API_KEY = get_secret("SENDGRID_API_KEY")
SENDGRID_FROM_EMAIL = get_secret("SENDGRID_FROM_EMAIL")

# ------------------------------
# Front-end URLs for deep links
# ------------------------------
# IMPORTANT: After you deploy, change this URL in your secrets to your live app's URL.
FRONTEND_BASE_URL = get_secret("FRONTEND_BASE_URL", "http://localhost:8501").rstrip("/")
VERIFY_LINK_TPL = f"{FRONTEND_BASE_URL}?verify={{token}}"
RESET_LINK_TPL  = f"{FRONTEND_BASE_URL}?reset={{token}}"