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


# ---------------------------------------------------------------------------
#  LLM PROVIDERS
# ---------------------------------------------------------------------------
def get_llm_provider(provider: str = "groq", model_name: str = "llama-3.1-8b-instant", temperature: float = 0.3):
    """
    Returns a LangChain-compatible LLM from a specific provider.
    """

    if provider == "groq":
        if not GROQ_API_KEY:
            print("Warning: GROQ_API_KEY not set. Using Dummy LLM.")
        else:
            return ChatGroq(
                groq_api_key=GROQ_API_KEY,
                model_name=model_name,
                temperature=temperature,
            )

    if provider == "gemini":
        if not GEMINI_API_KEY:
            print("Warning: GEMINI_API_KEY not set. Using Dummy LLM.")
        else:
            return ChatGoogleGenerativeAI(
                model=model_name,
                temperature=temperature,
                google_api_key=GEMINI_API_KEY,
            )

    # fallback dummy LLM
    from langchain.schema.messages import AIMessage

    class DummyLLM:
        def invoke(self, *args, **kwargs):
            return AIMessage(content="(No LLM configured — please set API keys.)")

    return DummyLLM()


# ---------------------------------------------------------------------------
#  MOOD CLASSIFICATION CHAIN (Updated for LC 1.1+)
# ---------------------------------------------------------------------------
def get_mood_extractor_chain():
    """
    Simple chain that extracts a single mood word.
    Uses new LangChain runnable pattern: prompt | llm
    """

    llm_classifier = get_llm_provider(
        provider="gemini",
        model_name="gemini-2.5-flash",
        temperature=0.0,
    )

    extractor_prompt = PromptTemplate.from_template(
        "Analyze the user's message. Identify the primary mood being expressed. "
        "Respond with a single word from this list: [happy, sad, anxious, angry, content, stressed, neutral]. "
        "If no clear mood is stated, respond with 'None'.\n\n"
        "User message: {input}"
    )

    chain = extractor_prompt | llm_classifier
    return chain


# ---------------------------------------------------------------------------
#  CRISIS KEYWORD CHECK
# ---------------------------------------------------------------------------
CRISIS_KEYWORDS = [
    "suicide", "kill myself", "end my life", "self-harm",
    "hurt myself", "want to die", "i'm going to die"
]

def contains_crisis_keywords(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in CRISIS_KEYWORDS)


# ---------------------------------------------------------------------------
#  LOGGING SETUP FOR EMAIL
# ---------------------------------------------------------------------------
logger = logging.getLogger("synermind.email")
if not logger.handlers:
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler("email.log")
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)


# ---------------------------------------------------------------------------
#  MOOD INSIGHTS + PLOTTING
# ---------------------------------------------------------------------------
def get_mood_insights_data(user_id):
    """
    Fetches mood log data for a user and returns a DataFrame.
    Converts timestamps to IST.
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
        {"timestamp": r.created_at, "mood": (r.mood.capitalize() if r.mood else None), "intensity": r.intensity}
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
    Creates a plotly line chart of average mood intensity per day.
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


# ---------------------------------------------------------------------------
#  EMAIL SENDER (SendGrid)
# ---------------------------------------------------------------------------
def send_email(to_email: str, subject: str, body: str) -> Dict[str, Any]:

    def _looks_like_email(s: str) -> bool:
        return isinstance(s, str) and bool(re.match(r"[^@\s]+@[^@\s]+\.[^@\s]+", s))

    if not _looks_like_email(to_email):
        return {"ok": False, "error": "Invalid email address."}

    if SENDGRID_API_KEY and SENDGRID_FROM_EMAIL:
        try:
            from sendgrid import SendGridAPIClient
            from sendgrid.helpers.mail import Mail

            body_with_breaks = body.replace('\n', '<br>')
            html_body = f"<strong>{body_with_breaks}</strong>"

            message = Mail(
                from_email=SENDGRID_FROM_EMAIL,
                to_emails=to_email,
                subject=subject,
                html_content=html_body
            )

            sg = SendGridAPIClient(SENDGRID_API_KEY)
            response = sg.send(message)

            if 200 <= response.status_code < 300:
                return {"ok": True}

            return {"ok": False, "error": f"SendGrid error {response.status_code}"}

        except Exception as e:
            logger.exception("Email failed:", exc_info=True)
            return {"ok": False, "error": str(e)}

    return {"ok": False, "error": "SendGrid API key or FROM_EMAIL missing."}


# ---------------------------------------------------------------------------
#  LANGCHAIN TOOL FUNCTIONS
# ---------------------------------------------------------------------------
def tool_log_mood(args: str) -> str:
    try:
        cleaned = args.strip().strip("'\"")
        parts = re.split(r'\\n|\n', cleaned)

        if len(parts) < 2:
            return f"ERROR: Input requires user id + mood."

        from db_models import resolve_user_identifier, add_mood

        try:
            user_id = resolve_user_identifier(parts[0].strip())
        except Exception:
            return f"ERROR: Unknown user: {parts[0].strip()}"

        mood = parts[1].strip()
        intensity = int(parts[2]) if len(parts) > 2 and parts[2].isdigit() else 5
        note = parts[3].strip() if len(parts) > 3 else None

        add_mood(user_id=user_id, mood=mood, intensity=intensity, note=note)
        return f"OK: Mood '{mood}' logged."

    except Exception as e:
        return f"ERROR: {str(e)}"


def tool_get_mood_history(args: str) -> str:
    try:
        from db_models import get_mood_history, resolve_user_identifier

        uid = resolve_user_identifier(args.strip())
        rows = get_mood_history(uid)

        if not rows:
            return "No mood history."

        return "\n".join(
            f"On {r.created_at.strftime('%Y-%m-%d')}, mood was '{r.mood}' (intensity {r.intensity})"
            for r in rows
        )

    except Exception as e:
        return f"ERROR: {str(e)}"


def tool_send_alert(args: str) -> str:
    try:
        from db_models import create_alert, SessionLocal, User, resolve_user_identifier

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
            return f"ALERT saved (id={alert.id}) but no valid recipient email."

        res = send_email(to_email, f"Synermind Alert: {subject}", f"User: {user.username}\n\n{message}")

        if res.get("ok"):
            return "ALERT sent successfully."

        return f"Alert saved but email failed: {res.get('error')}"

    except Exception as e:
        return f"ERROR: {str(e)}"


# ---------------------------------------------------------------------------
#  TOOL OBJECTS (correct for LC 1.1)
# ---------------------------------------------------------------------------
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
