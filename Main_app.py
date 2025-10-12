# Main_app.py
import streamlit as st
import time
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv() # Ensure this is at the very top

# Core DB & agents imports (safe baseline)
from db_models import (
    init_db,
    create_user,
    authenticate_user,
    log_interaction,
    add_mood,
    get_mood_history,
    SessionLocal,
    Interaction,
    User,
    log_feedback
)

# --- THIS IS THE FIX ---
# The old, incorrect 'mood_history_figure' has been removed from this import.
from llm_tools import contains_crisis_keywords, send_email
# --- END OF FIX ---

# Config (with graceful fallbacks if new constants aren't present)
try:
    from config import SECRET_KEY, VERIFY_LINK_TPL, RESET_LINK_TPL
except Exception:
    try:
        from config import SECRET_KEY
    except Exception:
        SECRET_KEY = "dev-secret"
    VERIFY_LINK_TPL = "http://localhost:8501?verify={token}"
    RESET_LINK_TPL = "http://localhost:8501?reset={token}"

# ---------- Page Setup ----------
st.set_page_config(page_title="Synermind — Multi-Agent Mental Wellness", layout="wide")
init_db()

# ---------- Session State Defaults ----------
if "user" not in st.session_state:
    st.session_state.user = None
if "auth_mode" not in st.session_state:
    st.session_state.auth_mode = "Sign In"

# Helper: robust query param getter
def _get_params():
    try:
        p = st.query_params
        return dict(p)
    except Exception:
        p = st.experimental_get_query_params()
        return {k: (v[0] if isinstance(v, list) and v else v) for k, v in p.items()}

params = _get_params()

# --- (The rest of your Main_app.py file is completely correct and remains unchanged) ---

# ---------- Handle Query Params (email verification / reset) ----------
if "verify" in params and params.get("verify"):
    try:
        from db_models import verify_email
        if verify_email(params.get("verify")):
            st.toast("Email verified successfully. You can sign in now.", icon="✅")
            st.session_state.auth_mode = "Sign In"
        else:
            st.toast("Verification link is invalid or expired.", icon="⚠️")
    except Exception:
        st.toast("Verification link processing failed.", icon="⚠️")

if "reset" in params and params.get("reset"):
    st.session_state["__show_reset_panel"] = True
    st.session_state["__reset_token"] = params.get("reset")

if "__post_signup_auth_mode" in st.session_state:
    st.session_state.auth_mode = st.session_state["__post_signup_auth_mode"]
    del st.session_state["__post_signup_auth_mode"]

# ---------- Global Styles (modern calm theme) ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Merriweather:wght@300;400;700&display=swap');
:root{
    --bg-1: #f4f9fb; /* very light blue */
    --bg-2: rgba(255,255,255,0.65);
    --accent: #6aa6d6; /* calm blue */
    --muted: #6b7280;
    --glass: rgba(255,255,255,0.55);
    --agent-bg: rgba(240,248,255,0.75);
    --user-bg: rgba(212,237,211,0.8);
    --card-radius: 14px;
}
html, body, .stApp {
    height: 100%;
    background: linear-gradient(180deg, #eef7fb 0%, #f8fafc 60%);
    font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
    color: #0f172a;
}

/* Glass card for main content */
.entry-wrap, .card {
    background: linear-gradient(180deg, rgba(255,255,255,0.7), rgba(255,255,255,0.5));
    backdrop-filter: blur(8px) saturate(120%);
    border-radius: var(--card-radius);
    border: 1px solid rgba(255,255,255,0.6);
    box-shadow: 0 6px 30px rgba(16,24,40,0.08);
}

.app-title{ font-family: 'Merriweather', serif; font-size: 1.6rem; font-weight:700; color: #0b3a57 }

/* Chat bubble styles */
.chat-wrapper{ padding: 12px; }
.chat-row{ margin-bottom: 10px; display:flex; align-items:flex-start; }
.chat-bubble{ max-width:640px; padding:12px 16px; border-radius:12px; font-size:0.98rem; line-height:1.4; box-shadow: 0 4px 18px rgba(11,26,40,0.06);}
.chat-bubble.agent{ background: linear-gradient(180deg, rgba(255,255,255,0.8), var(--agent-bg)); border-left:4px solid rgba(106,166,214,0.7); color:#07263a; border-top-left-radius:6px; border-bottom-left-radius:6px;}
.chat-bubble.user{ background: linear-gradient(180deg, rgba(255,255,255,0.8), var(--user-bg)); border-right:4px solid rgba(29,155,88,0.7); color:#03310b; border-top-right-radius:6px; border-bottom-right-radius:6px;}

/* Left / right column helpers */
.left-col{ display:flex; justify-content:flex-start; }
.right-col{ display:flex; justify-content:flex-end; }

/* Minor UI touches */
.stSidebar { background: linear-gradient(180deg, rgba(255,255,255,0.7), rgba(245,250,255,0.6)); }
.section-title{ font-weight:600; color:var(--accent); }

/* Make code blocks and links gentler */
pre, code{ background: rgba(15,23,42,0.03); padding:6px; border-radius:8px; }
a{ color:var(--accent); }

/* Responsive tweaks */
@media (max-width: 800px){ .chat-bubble{ max-width: 90%; } }
</style>
""", unsafe_allow_html=True)

# ---------- Auth Landing (centered) ----------
def render_auth_landing():
    # ... (This entire section of your code is correct and remains unchanged) ...
    # ... (It handles Sign In, Sign Up, MFA, Forgot Password, etc.) ...
    st.markdown('<div class="entry-wrap"><div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="app-title">Synermind — Multi‑Agent Mental Wellness</div>', unsafe_allow_html=True)

    if "__flash_msg" in st.session_state:
        st.success(st.session_state["__flash_msg"])
        del st.session_state["__flash_msg"]

    options = ["Sign In", "Sign Up"]
    default_choice = st.session_state.get("auth_mode", "Sign In")
    choice = st.segmented_control(
        label="Authentication Mode",
        options=options,
        default=default_choice,
        key="auth_mode_segment",
        label_visibility = "collapsed"
    )
    st.session_state.auth_mode = choice
    st.write("")

    if choice == "Sign In":
        st.markdown('<div class="section-title">Sign In</div>', unsafe_allow_html=True)
        with st.form("sign_in_form", clear_on_submit=True):
            si_username = st.text_input("Username", key="si_username", placeholder="Enter your username")
            si_password = st.text_input("Password", key="si_password", type="password", placeholder="Enter your password")
            c1, c2 = st.columns([1, 1])
            with c1:
                sign_in_submitted = st.form_submit_button("Sign In")
            with c2:
                forgot = st.form_submit_button("Forgot password?")

        if forgot:
            st.session_state["__show_forgot_panel"] = True

        if sign_in_submitted:
            try:
                from db_models import get_user_by_username, record_login_failure, record_login_success
                db_user = get_user_by_username(si_username.strip())
                if db_user and db_user.locked_until and db_user.locked_until > datetime.now(timezone.utc):
                    st.error("Account temporarily locked due to multiple failed attempts. Try again later.")
                else:
                    user = authenticate_user(si_username.strip(), si_password)
                    if user:
                        if getattr(user, "mfa_enabled", False) and getattr(user, "mfa_secret", None):
                            st.session_state["__mfa_user_id"] = user.id
                            st.session_state["__mfa_username"] = user.username
                            st.session_state["__pending_password_auth"] = True
                            st.rerun()
                        else:
                            record_login_success(user.id)
                            try:
                                from db_models import record_login
                                record_login(user.username)
                            except Exception:
                                pass
                            st.session_state.user = {"id": user.id, "username": user.username}
                            st.rerun()
                    else:
                        record_login_failure(si_username.strip())
                        st.error("Invalid credentials. Please check your username or password.")
            except Exception as e:
                st.error(f"Sign-in failed: {e}")

        if st.session_state.get("__pending_password_auth"):
            st.info("Enter the 6‑digit code from your authenticator app.")
            with st.form("mfa_form", clear_on_submit=True):
                otp = st.text_input("Authenticator code", max_chars=6)
                ok = st.form_submit_button("Verify")
            if ok:
                try:
                    from db_models import get_user_by_username, record_login_success
                    from security import verify_totp
                    uid = st.session_state.get("__mfa_user_id")
                    uname = st.session_state.get("__mfa_username")
                    u = get_user_by_username(uname)
                    if u and verify_totp(u.mfa_secret, otp):
                        record_login_success(u.id)
                        try:
                            from db_models import record_login
                            record_login(u.username)
                        except Exception:
                            pass
                        st.session_state.user = {"id": u.id, "username": u.username}
                        for k in ["__pending_password_auth", "__mfa_user_id", "__mfa_username"]:
                            st.session_state.pop(k, None)
                        st.rerun()
                    else:
                        st.error("Invalid code. Please try again.")
                except Exception as e:
                    st.error(f"MFA verification failed: {e}")

        if st.session_state.get("__show_forgot_panel"):
            st.markdown("#### Reset your password")
            with st.form("forgot_form", clear_on_submit=True):
                fp_email = st.text_input("Your account email")
                send = st.form_submit_button("Send reset link")
            if send:
                try:
                    from db_models import request_password_reset, get_reset_token
                    if request_password_reset(fp_email.strip()):
                        token_exp = get_reset_token(fp_email.strip())
                        if token_exp:
                            token, exp = token_exp
                            link = RESET_LINK_TPL.format(token=token)
                            res = send_email(
                                fp_email.strip(),
                                "Reset your Synermind password",
                                f"Click to reset your password:\n{link}\n\nThis link expires in 2 hours."
                            )
                            if res.get("ok"):
                                st.success("Reset link sent. Check your inbox.")
                            else:
                                st.warning("Email not configured. Use this one-time reset link:")
                                st.code(link)
                            st.session_state["__show_forgot_panel"] = False
                        else:
                            st.error("Could not create reset token. Try again.")
                    else:
                        st.error("No account found with that email.")
                except Exception as e:
                    st.error(f"Reset request failed: {e}")

        if st.session_state.get("__show_reset_panel"):
            st.markdown("#### Create a new password")
            with st.form("reset_form", clear_on_submit=True):
                npw = st.text_input("New password", type="password")
                npw2 = st.text_input("Confirm new password", type="password")
                ok = st.form_submit_button("Reset password")
            if ok:
                if npw != npw2 or len(npw) < 8:
                    st.error("Passwords must match and be at least 8 characters.")
                else:
                    try:
                        from db_models import reset_password
                        token = st.session_state.get("__reset_token")
                        if token and reset_password(token, npw):
                            st.success("Password updated. You can sign in now.")
                            st.session_state["__show_reset_panel"] = False
                            st.session_state["__reset_token"] = None
                        else:
                            st.error("Reset link is invalid or expired.")
                    except Exception as e:
                        st.error(f"Reset failed: {e}")
    else:  # Sign Up
        st.markdown('<div class="section-title">Sign Up</div>', unsafe_allow_html=True)
        with st.form("sign_up_form", clear_on_submit=True):
            su_username  = st.text_input("Username", key="su_username", placeholder="Create a username")
            su_email     = st.text_input("Email", key="su_email", placeholder="name@example.com")
            su_emergency = st.text_input("Emergency contact (email or phone)", key="su_emergency", placeholder="friend@example.com")
            su_password  = st.text_input("Password", key="su_password", type="password", placeholder="Create a strong password")
            
            agree = st.checkbox("I agree to the Terms of Service and Privacy Policy", value=False)
            sign_up_submitted = st.form_submit_button("Create Account")

        if sign_up_submitted:
            if not su_username.strip() or not su_email.strip() or not su_password:
                st.error("Username, Email, and Password are required.")
            elif not agree:
                st.error("You need to accept the Terms and Privacy to continue.")
            else:
                user = create_user(su_username.strip(), su_password, su_email.strip(), su_emergency.strip())
                if user:
                    try:
                        from db_models import SessionLocal, User, set_verification_token
                        db = SessionLocal()
                        try:
                            u = db.query(User).filter_by(id=user.id).first()
                            u.accepted_terms_at = datetime.now(timezone.utc)
                            db.commit()
                        finally:
                            db.close()

                        vtoken = set_verification_token(user.id)
                        if vtoken:
                            link = VERIFY_LINK_TPL.format(token=vtoken)
                            res = send_email(
                                user.email,
                                "Verify your Synermind email",
                                f"Welcome to Synermind!\n\nPlease verify your email by clicking:\n{link}\n\nThank you."
                            )
                            if not res.get("ok"):
                                st.warning("Email not configured. Use this one-time link to verify your email:")
                                st.code(link)
                    except Exception as e:
                        st.warning(f"Account created, but sending verification email failed: {e}")

                    st.session_state["__flash_msg"] = "Account created. Check your inbox to verify your email, then sign in."
                    st.session_state["__post_signup_auth_mode"] = "Sign In"
                    st.rerun()
                else:
                    st.error("Username already exists. Try a different one.")
    st.markdown('</div></div>', unsafe_allow_html=True)

# ---------- Entry ----------
from main_ui import render_main_ui

if st.session_state.user is None:
    render_auth_landing()
else:
    render_main_ui()