# llm_tools.py
import os
import re
import logging
import pandas as pd
import plotly.express as px
from typing import Dict, Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from config import GEMINI_API_KEY, GROQ_API_KEY, SENDGRID_API_KEY, SENDGRID_FROM_EMAIL

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

    from langchain.schema.messages import AIMessage
    class DummyLLM:
        def invoke(self, *args, **kwargs):
            return AIMessage(content="(No LLM configured — set GEMINI_API_KEY and/or GROQ_API_KEY.)")
    return DummyLLM()

def get_mood_extractor_chain():
    """
    Creates a simple, fast, and reliable chain whose ONLY job is to extract a mood.
    It uses Gemini for higher accuracy in this critical classification task.
    """
    llm_classifier = get_llm_provider(provider="gemini", model_name="gemini-2.5-flash", temperature=0.0)
    
    extractor_prompt = PromptTemplate.from_template(
        "Analyze the user's message. Identify the primary mood being expressed. "
        "Respond with a single word from this list: [happy, sad, anxious, angry, content, stressed, neutral]. "
        "If no clear mood is stated, respond with the single word 'None'. "
        "Do not add any other words or punctuation.\n\n"
        "User message: {input}"
    )
    return LLMChain(llm=llm_classifier, prompt=extractor_prompt, verbose=False)


CRISIS_KEYWORDS = [
    "suicide", "kill myself", "end my life", "self-harm",
    "hurt myself", "want to die", "i'm going to die"
]

def contains_crisis_keywords(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in CRISIS_KEYWORDS)


# Setup a simple file logger for email operations
logger = logging.getLogger("synermind.email")
if not logger.handlers:
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler("email.log")
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)


# --- THIS SECTION IS THE FIX ---

def get_mood_insights_data(user_id):
    """
    Fetches all mood log data for a user and returns it as a Pandas DataFrame.
    Timestamps are converted to IST (Asia/Kolkata) and date/time columns are provided.
    Accepts username or numeric id (resolve_user_identifier).
    """
    from db_models import get_mood_history, resolve_user_identifier  # local import
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

    # Ensure timestamp is datetime, localize naive as UTC, convert to IST
    ist = pytz.timezone("Asia/Kolkata")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
    def _to_ist(ts):
        if pd.isna(ts):
            return pd.NaT
        # if tz-naive assume UTC
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts.tz_convert(ist)
    df["timestamp_ist"] = df["timestamp"].apply(_to_ist)

    # date and time columns for display
    df["date"] = df["timestamp_ist"].dt.date
    df["time"] = df["timestamp_ist"].dt.strftime("%I:%M %p")  # 12-hour format

    return df

def plot_mood_trend_graph(df: pd.DataFrame):
    """
    Takes a DataFrame of mood data and returns a clean, readable Plotly line chart
    of the average mood intensity per day.
    """
    if df is None or df.empty:
        return None

    # Aggregate the data to get the average intensity for each date
    agg_df = df.groupby('date')['intensity'].mean().reset_index()

    # Create the figure
    fig = px.line(
        agg_df,
        x='date',
        y='intensity',
        title="Your Average Mood Intensity Over Time",
        markers=True,
        labels={'date': 'Date', 'intensity': 'Average Intensity'}
    )

    # --- FIX: Make the X-axis readable ---
    fig.update_xaxes(
        dtick="D1",  # Set ticks to appear one per day
        tickformat="%b %d\n%Y" # Format as "Oct 06\n2025"
    )
    fig.update_layout(
        title_font_size=20,
        xaxis_title=None,
    )
    return fig

# --- END OF FIX ---


def send_email(to_email: str, subject: str, body: str) -> Dict[str, Any]:
    """
    Send email using SendGrid API. This version is more robust and compatible with Python 3.10.
    """
    # basic recipient validation: must contain @ and a dot
    def _looks_like_email(s: str) -> bool:
        if not s or not isinstance(s, str):
            return False
        return bool(re.match(r"[^@\s]+@[^@\s]+\.[^@\s]+", s))

    if not _looks_like_email(to_email):
        logger.warning("Attempted to send email to non-email recipient: %s", to_email)
        return {"ok": False, "error": "Recipient does not appear to be an email address."}

    if SENDGRID_API_KEY and SENDGRID_FROM_EMAIL:
        try:
            from sendgrid import SendGridAPIClient
            from sendgrid.helpers.mail import Mail

            # --- THE FIX: Perform the replacement BEFORE the f-string ---
            # This avoids the backslash syntax error in Python 3.10
            body_with_breaks = body.replace('\n', '<br>')
            html_body = f"<strong>{body_with_breaks}</strong>"
            
            message = Mail(
                from_email=SENDGRID_FROM_EMAIL,
                to_emails=to_email,
                subject=subject,
                html_content=html_body  # Use the new, clean variable
            )
            
            sg = SendGridAPIClient(SENDGRID_API_KEY)
            response = sg.send(message)
            logger.debug("SendGrid response - status: %s, body: %s", getattr(response, 'status_code', None), getattr(response, 'body', None))

            if 200 <= getattr(response, 'status_code', 0) < 300:
                logger.info("Email sent to %s (subject=%s)", to_email, subject)
                return {"ok": True}
            else:
                err = f"SendGrid API error ({getattr(response, 'status_code', 'unknown')}): {getattr(response, 'body', '')}"
                logger.error("Failed to send email: %s", err)
                return {"ok": False, "error": err}

        except ImportError:
             return {"ok": False, "error": "The 'sendgrid' library is not installed. Please run 'pip install sendgrid'."}
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logger.exception("Exception while sending email to %s: %s", to_email, tb)
            return {"ok": False, "error": f"A critical error occurred while sending email: {str(e)}"}
    
    return {"ok": False, "error": "SendGrid API Key or From Email is not configured in your .env file."}

def tool_log_mood(args: str) -> str:
    """
    Parses a multi-line string from the LLM to log a user's mood.
    This function is resilient to any kind of newline formatting from the LLM.
    """
    try:
        cleaned_args = args.strip().strip("'\"")
        parts = re.split(r'\\n|\n', cleaned_args)
        if len(parts) < 2:
            return f"ERROR: Input must contain user identifier and mood. Received: {parts}"
        # Allow username or numeric id
        from db_models import resolve_user_identifier, add_mood
        try:
            user_id = resolve_user_identifier(parts[0].strip())
        except Exception:
            return f"ERROR: Could not resolve user identifier: {parts[0].strip()}"
        mood = parts[1].strip()
        intensity = int(parts[2].strip()) if len(parts) > 2 and parts[2].strip().isdigit() else 5
        note = parts[3].strip() if len(parts) > 3 else None
        ml = add_mood(user_id=user_id, mood=mood, intensity=intensity, note=note)
        return f"OK: Mood '{mood}' was successfully logged for user {user_id}."
    except Exception as e:
        return f"ERROR logging mood: {str(e)}. Input received: {args}"

def tool_get_mood_history(args: str) -> str:
    from db_models import get_mood_history, resolve_user_identifier
    try:
        uid = resolve_user_identifier(args.strip())
        rows = get_mood_history(uid)
        lines = [f"On {r.created_at.strftime('%Y-%m-%d')}, mood was '{r.mood}' (intensity: {r.intensity})" for r in rows]
        return "\n".join(lines) if lines else "No mood history found for this user."
    except Exception as e:
        return f"ERROR getting mood history: {str(e)}"

def tool_send_alert(args: str) -> str:
    from db_models import create_alert, SessionLocal, User, resolve_user_identifier
    try:
        parts = args.split("\n", 2)
        uid = resolve_user_identifier(parts[0].strip())
        subject = parts[1]
        message = parts[2] if len(parts) > 2 else ""
        a = create_alert(user_id=uid, alert_type=subject, message=message)
        db = SessionLocal()
        user = db.query(User).filter(User.id == uid).first()
        db.close()
        to_email = None
        if user:
            # Prefer emergency_contact if it looks like an email, otherwise fallback to the user's email
            ec = user.emergency_contact.strip() if user.emergency_contact else None
            ue = user.email.strip() if user.email else None
            # re-use send_email's internal validation by checking basic pattern here
            if ec and re.match(r"[^@\s]+@[^@\s]+\.[^@\s]+", ec):
                to_email = ec
            elif ue and re.match(r"[^@\s]+@[^@\s]+\.[^@\s]+", ue):
                to_email = ue

        if not to_email:
            logger.warning("Alert saved (id=%s) but no valid recipient email found for user id %s", a.id, uid)
            return f"ALERT saved (id={a.id}) but no valid recipient email found for this user."

        res = send_email(to_email, f"Synermind Alert: {subject}", f"This is an alert regarding user: {user.username}\n\n{message}")
        if res.get("ok"):
            return f"ALERT sent successfully to the user's emergency contact."
        return f"Alert was saved, but the email failed to send: {res.get('error')}"
    except Exception as e:
        return f"ERROR sending alert: {str(e)}"

# --- LangChain Tool Objects ---
LOG_MOOD_TOOL = Tool.from_function(func=tool_log_mood, name="log_mood", description="Logs a user's current mood.")
GET_MOOD_HISTORY_TOOL = Tool.from_function(func=tool_get_mood_history, name="get_mood_history", description="Retrieves the mood history for a user.")
SEND_ALERT_TOOL = Tool.from_function(func=tool_send_alert, name="send_alert", description="Sends a crisis alert.")