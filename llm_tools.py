# llm_tools.py
import os
import re
import logging
import pandas as pd
import plotly.express as px
from typing import Dict, Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate

from config import GEMINI_API_KEY, GROQ_API_KEY, SENDGRID_API_KEY, SENDGRID_FROM_EMAIL


# -----------------------------------------------------------------------------
# LLM PROVIDERS
# -----------------------------------------------------------------------------

def get_llm_provider(provider: str = "groq", model_name: str = "llama-3.1-8b-instant", temperature: float = 0.3):
    """
    Returns a LangChain-compatible LLM from a specific provider.
    """
    if provider == "groq":
        api_key = GROQ_API_KEY
        if not api_key:
            print("Warning: GROQ_API_KEY not set. Using fallback.")
        else:
            return ChatGroq(
                temperature=temperature,
                groq_api_key=api_key,
                model_name=model_name,
            )

    if provider == "gemini":
        api_key = GEMINI_API_KEY
        if not api_key:
            print("Warning: GEMINI_API_KEY not set. Using fallback.")
        else:
            return ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                google_api_key=api_key,
            )

    # fallback dummy
    from langchain.schema.messages import AIMessage

    class DummyLLM:
        def invoke(self, *args, **kwargs):
            return AIMessage(content="(No LLM configured — set GEMINI_API_KEY and/or GROQ_API_KEY.)")

    return DummyLLM()


# -----------------------------------------------------------------------------
# MOOD CLASSIFIER — FIXED (removed LLMChain)
# -----------------------------------------------------------------------------

def get_mood_extractor_chain():
    """
    Creates a simple chain whose ONLY job is to extract a mood.
    Now uses the new Runnable pipeline instead of removed LLMChain.
    """
    llm_classifier = get_llm_provider(
        provider="gemini",
        model_name="gemini-2.5-flash",
        temperature=0.0
    )

    extractor_prompt = PromptTemplate.from_template(
        "Analyze the user's message. Identify the primary mood being expressed. "
        "Respond with a single word from this list: [happy, sad, anxious, angry, content, stressed, neutral]. "
        "If no clear mood is stated, respond with the single word 'None'. "
        "Do not add any other words or punctuation.\n\n"
        "User message: {input}"
    )

    # The new chain: prompt → llm
    chain = extractor_prompt | llm_classifier
    return chain


# -----------------------------------------------------------------------------
# CRISIS DETECTION
# -----------------------------------------------------------------------------

CRISIS_KEYWORDS = [
    "suicide", "kill myself", "end my life", "self-harm",
    "hurt myself", "want to die", "i'm going to die"
]

def contains_crisis_keywords(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in CRISIS_KEYWORDS)


# -----------------------------------------------------------------------------
# EMAIL LOGGER
# -----------------------------------------------------------------------------

logger = logging.getLogger("synermind.email")
if not logger.handlers:
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler("email.log")
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)


# -----------------------------------------------------------------------------
# MOOD INSIGHTS + GRAPH
# -----------------------------------------------------------------------------

def get_mood_insights_data(user_id):
    """
    Fetches mood log data and returns a Pandas DataFrame with IST timestamps.
    """
    from db_models import get_mood_history, resolve_user_identifier
    import pytz

    try:
        uid = resolve_user_identifier(user_id)
    except Exception:
        return None

    rows = get_mood_history(uid)
    if not rows:
        return None

    df = pd.DataFrame([
        {
            "timestamp": r.created_at,
            "mood": r.mood.capitalize() if r.mood else None,
            "intensity": r.intensity
        }
        for r in rows
    ])

    if df.empty:
        return None

    ist = pytz.timezone("Asia/Kolkata")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')

    def _to_ist(ts):
        if pd.isna(ts):
            return pd.NaT
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts.tz_convert(ist)

    df["timestamp_ist"] = df["timestamp"].apply(_to_ist)
    df["date"] = df["timestamp_ist"].dt.date
    df["time"] = df["timestamp_ist"].dt.strftime("%I:%M %p")

    return df


def plot_mood_trend_graph(df: pd.DataFrame):
    """
    Plot average mood intensity per day.
    """
    if df is None or df.empty:
        return None

    agg_df = df.groupby('date')['intensity'].mean().reset_index()

    fig = px.line(
        agg_df,
        x='date',
        y='intensity',
        title="Your Average Mood Intensity Over Time",
        markers=True,
        labels={'date': 'Date', 'intensity': 'Average Intensity'}
    )

    fig.update_xaxes(
        dtick="D1",
        tickformat="%b %d\n%Y"
    )
    fig.update_layout(
        title_font_size=20,
        xaxis_title=None,
    )
    return fig


# -----------------------------------------------------------------------------
# SENDGRID EMAIL
# -----------------------------------------------------------------------------

def send_email(to_email: str, subject: str, body: str) -> Dict[str, Any]:
    """
    Sends an email via SendGrid with safer formatting.
    """
    def _looks_like_email(s: str) -> bool:
        return bool(s and isinstance(s, str) and re.match(r"[^@\s]+@[^@\s]+\.[^@\s]+", s))

    if not _looks_like_email(to_email):
        logger.warning("Invalid email: %s", to_email)
        return {"ok": False, "error": "Invalid email address."}

    if SENDGRID_API_KEY and SENDGRID_FROM_EMAIL:
        try:
            from sendgrid import SendGridAPIClient
            from sendgrid.helpers.mail import Mail

            body_html = f"<strong>{body.replace('\n', '<br>')}</strong>"

            message = Mail(
                from_email=SENDGRID_FROM_EMAIL,
                to_emails=to_email,
                subject=subject,
                html_content=body_html
            )

            sg = SendGridAPIClient(SENDGRID_API_KEY)
            response = sg.send(message)

            if 200 <= getattr(response, 'status_code', 0) < 300:
                return {"ok": True}

            return {"ok": False, "error": f"SendGrid error: {response.status_code}"}

        except Exception as e:
            return {"ok": False, "error": str(e)}

    return {"ok": False, "error": "SendGrid not configured."}


# -----------------------------------------------------------------------------
# TOOL FUNCTIONS
# -----------------------------------------------------------------------------

def tool_log_mood(args: str) -> str:
    """
    Parses LLM input to log mood.
    """
    try:
        cleaned_args = args.strip().strip("'\"")
        parts = re.split(r'\\n|\n', cleaned_args)

        if len(parts) < 2:
            return f"ERROR: Input must contain user identifier and mood."

        from db_models import resolve_user_identifier, add_mood

        user_id = resolve_user_identifier(parts[0].strip())
        mood = parts[1].strip()
        intensity = int(parts[2].strip()) if len(parts) > 2 and parts[2].strip().isdigit() else 5
        note = parts[3].strip() if len(parts) > 3 else None

        add_mood(user_id=user_id, mood=mood, intensity=intensity, note=note)

        return f"OK: Mood '{mood}' logged for user {user_id}."

    except Exception as e:
        return f"ERROR logging mood: {str(e)}"


def tool_get_mood_history(args: str) -> str:
    from db_models import get_mood_history, resolve_user_identifier
    try:
        uid = resolve_user_identifier(args.strip())
        rows = get_mood_history(uid)

        if not rows:
            return "No mood history found."

        lines = [
            f"On {r.created_at.strftime('%Y-%m-%d')}, mood was '{r.mood}' (intensity: {r.intensity})"
            for r in rows
        ]
        return "\n".join(lines)

    except Exception as e:
        return f"ERROR: {str(e)}"


def tool_send_alert(args: str) -> str:
    from db_models import create_alert, SessionLocal, User, resolve_user_identifier
    try:
        parts = args.split("\n", 2)
        uid = resolve_user_identifier(parts[0].strip())
        subject = parts[1]
        message = parts[2] if len(parts) > 2 else ""

        alert = create_alert(user_id=uid, alert_type=subject, message=message)

        db = SessionLocal()
        user = db.query(User).filter(User.id == uid).first()
        db.close()

        to_email = None
        if user:
            if user.emergency_contact and re.match(r"[^@\s]+@[^@\s]+\.[^@\s]+", user.emergency_contact):
                to_email = user.emergency_contact
            elif user.email and re.match(r"[^@\s]+@[^@\s]+\.[^@\s]+", user.email):
                to_email = user.email

        if not to_email:
            return f"ALERT saved (id={alert.id}) but no valid email found."

        res = send_email(
            to_email,
            f"Synermind Alert: {subject}",
            f"Alert regarding user: {user.username}\n\n{message}"
        )

        if res.get("ok"):
            return "ALERT sent successfully."
        else:
            return f"Alert saved but email failed: {res.get('error')}"

    except Exception as e:
        return f"ERROR sending alert: {str(e)}"


# -----------------------------------------------------------------------------
# TOOL OBJECTS
# -----------------------------------------------------------------------------

LOG_MOOD_TOOL = Tool.from_function(
    func=tool_log_mood,
    name="log_mood",
    description="Logs a user's current mood."
)

GET_MOOD_HISTORY_TOOL = Tool.from_function(
    func=tool_get_mood_history,
    name="get_mood_history",
    description="Retrieves mood history for a user."
)

SEND_ALERT_TOOL = Tool.from_function(
    func=tool_send_alert,
    name="send_alert",
    description="Sends a crisis alert."
)
