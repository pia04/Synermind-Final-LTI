# main_ui.py
import streamlit as st
import time
from datetime import datetime
import numpy as np
import pandas as pd # <-- The required import for the new dashboard

# --- All Necessary Imports from Your Project ---
from db_models import (
    SessionLocal,
    Interaction,
    User,
    log_feedback,
    add_mood,
    delete_user_interactions,
    set_mfa,
    get_user_metrics
)
from agents import get_agents
from router import router_chain
# --- FIX: Import the new functions and remove the old one ---
from llm_tools import get_mood_insights_data, plot_mood_trend_graph, get_mood_extractor_chain
from security import (
    new_totp_secret,
    totp_provisioning_uri,
    qr_png_data_uri,
    verify_totp
)

# Use Streamlit's cache to create agents once per user session
@st.cache_resource
def load_agents_for_session():
    return get_agents()

@st.cache_resource
def load_extractor_chain():
    return get_mood_extractor_chain()

def load_chat_history(user_id: int):
    """Loads chat history for a specific user from the database."""
    db = SessionLocal()
    try:
        rows = (
            db.query(Interaction)
            .filter(Interaction.user_id == user_id, Interaction.agent_type != "feedback")
            .order_by(Interaction.created_at.asc())
            .all()
        )
        history = []
        for r in rows:
            history.append({"role": "user", "content": r.user_msg})
            history.append({"role": "assistant", "avatar": "🤖", "content": r.agent_reply, "agent": r.agent_type})
        return history
    finally:
        db.close()

def estimate_intensity_from_text(text: str) -> int:
    """Heuristic estimate 1-10 intensity from short user text (conservative)."""
    if not text:
        return 5
    t = text.lower()
    import re
    m = re.search(r'\b([1-9]|10)\b', t)
    if m:
        try:
            v = int(m.group(1))
            return max(1, min(10, v))
        except Exception:
            pass
    high = ["very", "extremely", "overwhelmed", "panic", "panic attack", "terrified", "intense", "severe", "horrible"]
    med_high = ["anxious", "anxiety", "stressed", "panic", "scared", "worried"]
    low = ["bit", "little", "slightly", "calm", "okay", "fine", "neutral"]
    score = 5
    for kw in high:
        if kw in t:
            score = max(score, 8)
    for kw in med_high:
        if kw in t:
            score = max(score, 6)
    for kw in low:
        if kw in t:
            score = min(score, 3)
    if "!!!" in t or "!!" in t:
        score = min(10, score + 2)
    if t.count("?") >= 2 and score < 8:
        score += 1
    return max(1, min(10, int(score)))

def render_main_ui():
    user = st.session_state.user

# --- ADD THIS BLOCK TO DISPLAY THE PERMANENT ERROR ---
    if 'crisis_error' in st.session_state:
        st.error(st.session_state['crisis_error'])
        # Clear the error so it doesn't show forever
        del st.session_state['crisis_error']
    # ---------------------------------------------------
    # --- Initialize Session State ---
    if "chat_ended" not in st.session_state:
        st.session_state.chat_ended = False
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = load_chat_history(user["id"])
    if "response_times" not in st.session_state:
        st.session_state.response_times = []

    # --- Sidebar Navigation ---
    st.sidebar.title("Navigation")
    st.sidebar.markdown(f"**Logged in as:** {user['username']}")

    if not st.session_state.chat_ended:
        if st.sidebar.button("🟥 End Chat Session"):
            st.session_state.chat_ended = True
            st.rerun()
    else:
        if st.sidebar.button("▶️ Start New Chat"):
            st.session_state.chat_ended = False
            st.session_state.chat_history = []
            st.session_state.last_agent_used = ""
            st.rerun()

    if st.sidebar.button("Logout"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()

    with st.sidebar.expander("⚠️ Manage History"):
        if st.button("Clear Chat History"):
            st.session_state.confirm_delete = True
        
        if st.session_state.get("confirm_delete"):
            st.warning("Are you sure you want to permanently delete your chat history?")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Yes, Delete It", type="primary"):
                    if delete_user_interactions(user["id"]):
                        st.session_state.chat_history = []
                        st.session_state.confirm_delete = False
                        st.success("History cleared.")
                        time.sleep(1); st.rerun()
                    else:
                        st.error("Could not clear history.")
            with c2:
                if st.button("Cancel"):
                    st.session_state.confirm_delete = False; st.rerun()

    page = st.sidebar.selectbox(
        "Go to",
        ("Chat", "Mood Logger", "Mood Insights", "Metrics & Insights", "Security (MFA)")
    )

    # --- Page Content ---
    if page == "Chat":
        st.title(f"Synermind Wellness Chat")

        st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)
        for message in st.session_state.chat_history:
            role = message.get("role")
            content = message.get("content")
            agent = message.get("agent", None)

            if role == 'assistant':
                # Agent on the right
                html = f'''
                <div class="chat-row">
                    <div class="right-col" style="width:100%">
                        <div class="chat-bubble agent">
                            {f"<div style='font-size:0.82em;color:#2b556a;margin-bottom:6px;font-weight:600;text-align:right;'>{agent.capitalize()} Agent</div>" if agent else ''}
                            {content}
                        </div>
                    </div>
                </div>
                '''
                st.markdown(html, unsafe_allow_html=True)
            else:
                # User on the left
                html = f'''
                <div class="chat-row">
                    <div class="left-col" style="width:100%">
                        <div class="chat-bubble user">
                            {content}
                        </div>
                    </div>
                </div>
                '''
                st.markdown(html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if not st.session_state.get("chat_ended"):
            AGENTS = load_agents_for_session()
            mood_extractor = load_extractor_chain()

            user_msg = st.chat_input("How are you feeling today?")
            if user_msg:
                st.session_state.chat_history.append({"role": "user", "content": user_msg})
                with st.chat_message("user"):
                    st.markdown(user_msg)

                # --- CRISIS BYPASS: If crisis keywords detected, send alert directly ---
                from llm_tools import contains_crisis_keywords, send_email
                if contains_crisis_keywords(user_msg):
                    import db_models
                    db = db_models.SessionLocal()
                    user_obj = db.query(db_models.User).filter(db_models.User.id == user["id"]).first()
                    db.close()

                    to_email = user_obj.emergency_contact if user_obj and user_obj.emergency_contact else None
                    
                    # Fallback to user's own email if emergency contact is blank
                    if not to_email and user_obj:
                        to_email = user_obj.email

                    if to_email:
                        # Attempt to send the email
                        db_models.create_alert(user_id=user["id"], alert_type="CRISIS ALERT: User expresses intent for self-harm", message=user_msg)
                        res = send_email(to_email, "Synermind Alert: CRISIS ALERT", f"This is an alert regarding user: {user_obj.username}\n\nUser message: {user_msg}")
                        
                        # --- THIS IS THE FIX ---
                        # If the email fails, store the error in session_state before rerunning
                        if not res.get("ok"):
                            st.session_state['crisis_error'] = f"The send_email function failed. Reason: {res.get('error')}"
                        # ---------------------

                    else:
                        st.session_state['crisis_error'] = "FINAL ERROR: No emergency contact OR primary email could be found for this user. Cannot send alert."
                    
                    # Display a safe message to the user and rerun the page
                    st.session_state.chat_history.append({"role": "assistant", "avatar": "🤖", "content": "It sounds like you are in distress. An alert has been dispatched to your emergency contact. Please reach out to a trusted person or a crisis hotline immediately.", "agent": "crisis"})
                    st.rerun()
                # --- Otherwise, normal agent flow ---
                with st.chat_message("assistant", avatar="🤖"):
                    with st.spinner("Thinking..."):
                        try:
                            extracted_mood = mood_extractor.run(user_msg).strip().lower()
                            if extracted_mood and extracted_mood != "none":
                                try:
                                    estimated_intensity = estimate_intensity_from_text(user_msg)
                                except Exception:
                                    estimated_intensity = 5
                                add_mood(user_id=user['id'], mood=extracted_mood,intensity = estimated_intensity)
                                st.toast(f"Mood logged: {extracted_mood.capitalize()} (intensity { estimated_intensity})", icon="📝")
                        except Exception as e:
                            print(f"Mood extraction failed: {e}")

                        import hashlib
                        def estimate_tokens(text: str) -> int:
                            if not text:
                                return 0
                            return max(1, int(len(text) / 4))
                        def trim_history_by_tokens(history, max_tokens=800):
                            lines = []
                            total = 0
                            for m in reversed(history):
                                line = f"{'Human' if m['role']=='user' else 'AI'}: {m['content']}"
                                t = estimate_tokens(line)
                                if total + t > max_tokens:
                                    break
                                lines.append(line)
                                total += t
                            return "\n".join(reversed(lines))
                        if "response_cache" not in st.session_state:
                            st.session_state.response_cache = {}
                        def make_cache_key(agent_label: str, user_msg: str, context_text: str):
                            h = hashlib.sha256()
                            h.update(agent_label.encode('utf-8'))
                            h.update(b"||")
                            h.update(user_msg.encode('utf-8'))
                            h.update(b"||")
                            h.update(context_text.encode('utf-8'))
                            return h.hexdigest()
                        recent_history = st.session_state.chat_history
                        context_text = trim_history_by_tokens(recent_history, max_tokens=800)
                        # Ensure agents (especially the Crisis agent) have the current user's identifier
                        # The Crisis agent's tool expects the ACTION INPUT to begin with the user identifier
                        # (username or numeric id). Prepend this info so the agent can call tools reliably.
                        user_ident_line = f"User-Identifier: {user['username']} (id:{user['id']})"
                        router_input = user_ident_line + "\n" + context_text + f"\nHuman: {user_msg}"
                        agent_label = router_chain.run(router_input)
                        last_agent = st.session_state.get("last_agent_used", "")
                        if agent_label != last_agent and last_agent != "":
                            st.toast(f"Switched to {agent_label.capitalize()} Agent", icon="🤖")
                        st.session_state.last_agent_used = agent_label
                        agent = AGENTS.get(agent_label, AGENTS["mood"])
                        MAX_RETRIES = 3
                        backoff = 1.0
                        response = None
                        start_time = time.time()
                        cache_key = make_cache_key(agent_label, user_msg, context_text)
                        cache_entry = st.session_state.response_cache.get(cache_key)
                        if cache_entry:
                            cached_response, cached_ts = cache_entry
                            if time.time() - cached_ts < 300:
                                response = cached_response
                        if response is None:
                            for attempt in range(1, MAX_RETRIES + 1):
                                try:
                                    prompt = f"{context_text}\nHuman: {user_msg}\nPlease be concise and practical in your reply (limit to 150 tokens)."
                                    response = agent.run(input=prompt)
                                    st.session_state.response_cache[cache_key] = (response, time.time())
                                    break
                                except Exception as e:
                                        err_text = str(e).lower()
                                        # Detect rate limits and retry as before
                                        if 'rate' in err_text and ('limit' in err_text or 'rate_limit' in err_text or 'rate-limit' in err_text):
                                            if attempt == MAX_RETRIES:
                                                st.error("The language model is temporarily busy due to rate limits. Please wait a moment and try again.")
                                                response = "I'm having trouble accessing the language model right now. Please try again shortly."
                                            else:
                                                time.sleep(backoff)
                                                backoff *= 2
                                                continue

                                        # Detect authentication errors (invalid API key / unauthorized)
                                        if ('invalid api key' in err_text) or ('invalid_api_key' in err_text) or ('unauthorized' in err_text) or ('authenticationerror' in err_text) or ('authentication error' in err_text):
                                            # Surface a user-friendly error in the UI and avoid crashing
                                            st.error("Language model authentication failed (invalid or missing API key). Please check your GROQ/GEMINI API configuration.")
                                            st.session_state['__llm_auth_error'] = err_text
                                            response = "I'm temporarily unable to access the language model due to configuration. Please notify the administrator or check the API keys."
                                            break

                                        # Unknown error: re-raise so it surfaces (developer will see full traceback)
                                        raise
                        end_time = time.time()
                        st.session_state.response_times.append(end_time - start_time)
                        from db_models import log_interaction
                        log_interaction(user_id=user['id'], agent_type=agent_label, user_msg=user_msg, agent_reply=response)
                        st.markdown(response)
                        st.session_state.chat_history.append({"role": "assistant", "avatar": "🤖", "content": response, "agent": agent_label})
                st.rerun()
        else:
            # Feedback UI...
            st.subheader("Thank you for chatting!")
            st.write("Your feedback helps us improve.")
            feedback_rating = st.slider("How helpful was this session?", 1, 5, 3)
            emojis = {1: "😔", 2: "😕", 3: "😐", 4: "🙂", 5: "😃"}
            st.markdown(f"<p style='text-align: center; font-size: 5rem;'>{emojis[feedback_rating]}</p>", unsafe_allow_html=True)
            feedback_comment = st.text_area("Any additional comments? (Optional)")
            if st.button("Submit Feedback"):
                log_feedback(user_id=user['id'], rating=feedback_rating, comment=feedback_comment)
                st.success("Feedback submitted! Thank you."); st.balloons(); time.sleep(2); st.rerun()

    elif page == "Mood Logger":
        st.header("Manual Mood Logger")
        st.info("You can also log your mood conversationally in the chat!")
        mood = st.selectbox("How are you feeling?", ["Happy", "Content", "Neutral", "Sad", "Anxious", "Angry", "Stressed"])
        intensity = st.slider("Intensity", 1, 10, 5)
        note = st.text_area("Optional note (e.g., 'After my meeting')", key="mood_note")
        if st.button("Log Mood"):
            add_mood(user_id=user['id'], mood=mood, intensity=intensity, note=note)
            st.success("Mood logged successfully.")
            st.toast("Your new entry is visible in Mood Insights!", icon="📊")

    # --- THIS IS THE NEW, UPGRADED MOOD INSIGHTS PAGE ---
    elif page == "Mood Insights":
        st.title("📊 Your Mood Insights Dashboard")
        
        # 1. Fetch all mood data once (pass username so resolver can map to id)
        df = get_mood_insights_data(user['username'])

        if df is None or df.empty:
            st.info("No mood data has been logged yet. Chat with the agents or use the Manual Mood Logger to see your trends.")
        else:
            # 2. Display the Trend Graph
            st.subheader("Your Mood Trend")
            trend_fig = plot_mood_trend_graph(df)
            st.plotly_chart(trend_fig, use_container_width=True)

            st.divider()

            # 3. Display the Detailed Mood Log
            st.subheader("Detailed Mood Log")
            
            # Prepare the DataFrame for display
            display_df = df[['date', 'time', 'mood', 'intensity']].copy()
            display_df.rename(
                columns={
                    'date': 'Date',
                    'time': 'Time',
                    'mood': 'Mood',
                    'intensity': 'Intensity (1-10)'
                },
                inplace=True
            )
            
            # Display the table, newest entries first
            st.dataframe(
                display_df.sort_values(by=['Date', 'Time'], ascending=[False, False]),
                use_container_width=True,
                hide_index=True
            )

    elif page == "Metrics & Insights":
        st.title("📊 Your Metrics & Insights")
        metrics = get_user_metrics(user['id'])
        st.header("Your Journey")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Conversation Streak", f"{metrics['conversation_streak']} Days")
        col2.metric("Total Logins Today", metrics['daily_logins'])
        col3.metric("Total Messages Sent", metrics['total_interactions'])
        col4.metric("Avg. Session Rating", f"{metrics['avg_feedback_rating']} / 5.0 ⭐")
        st.subheader("Agent Usage")
        st.write("This chart shows which specialist you've connected with the most.")
        if metrics['agent_usage']:
            st.bar_chart(metrics['agent_usage'])
        else:
            st.info("No agent conversations yet. Start a chat to see your usage patterns!")
        st.divider()
        st.header("System Performance")
        colA, colB = st.columns(2)
        with colA:
            st.subheader("Response Time (Latency)")
            if st.session_state.response_times:
                avg_response_time = np.mean(st.session_state.response_times)
                st.metric("Avg. Response Time (Current Session)", f"{avg_response_time:.2f} sec")
            else:
                st.info("No responses yet in this session. Latency will be measured as you chat.")
        with colB:
            st.subheader("Mood Logging")
            st.metric("Total Moods Logged Successfully", metrics['total_moods_logged'])
            st.write("This confirms the Mood Agent is successfully saving your check-ins.")

    elif page == "Security (MFA)":
        st.header("Multi-Factor Authentication (MFA)")
        db = SessionLocal()
        try:
            u = db.query(User).filter(User.id == user['id']).first()
            if u:
                if not u.mfa_enabled:
                    st.info("Enhance your account security by enabling MFA. You will need an authenticator app (like Google Authenticator or Authy).")
                    if st.button("Enable MFA"):
                        secret = new_totp_secret()
                        st.session_state["__pending_mfa_secret"] = secret
                        uri = totp_provisioning_uri(secret, u.username, "Synermind")
                        img = qr_png_data_uri(uri)
                        st.image(img, caption="1. Scan this QR code in your authenticator app")
                        st.code(secret, language=None)
                        st.markdown("2. Enter the 6-digit code from your app below to confirm.")
                    
                    if "__pending_mfa_secret" in st.session_state:
                        secret = st.session_state["__pending_mfa_secret"]
                        with st.form("confirm_mfa"):
                            code = st.text_input("6-Digit Code")
                            if st.form_submit_button("Confirm and Activate MFA"):
                                if verify_totp(secret, code):
                                    set_mfa(u.id, True, secret)
                                    del st.session_state["__pending_mfa_secret"]
                                    st.success("MFA has been enabled!")
                                    time.sleep(1); st.rerun()
                                else:
                                    st.error("Invalid code. Please try again.")
                else:
                    st.success("MFA is currently enabled on your account.")
                    if st.button("Disable MFA"):
                        st.session_state.confirm_disable_mfa = True
                    if st.session_state.get("confirm_disable_mfa"):
                        st.warning("Are you sure you want to disable MFA?")
                        with st.form("disable_mfa_form"):
                            password = st.text_input("Enter your password to confirm", type="password")
                            if st.form_submit_button("Yes, Disable MFA", type="primary"):
                                from db_models import authenticate_user
                                if authenticate_user(u.username, password):
                                    set_mfa(u.id, False, None)
                                    del st.session_state.confirm_disable_mfa
                                    st.success("MFA disabled.")
                                    time.sleep(1); st.rerun()
                                else:
                                    st.error("Incorrect password.")
        finally:
            db.close()