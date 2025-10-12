# db_models.py
from sqlalchemy import (
    create_engine, Column, Integer, String, DateTime, Text, Boolean, ForeignKey
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.sql import func
from sqlalchemy.exc import IntegrityError
from datetime import datetime, timedelta, timezone
from sqlalchemy import func, distinct
from datetime import date, timedelta
from config import DATABASE_URL
import bcrypt
import secrets

# ---------- SQLAlchemy setup ----------
Base = declarative_base()
engine = create_engine(DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

# ---------- Password Utilities ----------
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode()

def verify_password(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False

def _now():
    return datetime.now(timezone.utc)

def _new_token(n=24):
    return secrets.token_urlsafe(n)

# ---------- Models ----------
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(128), unique=True, index=True, nullable=False)
    password_hash = Column(String(256), nullable=False)
    email = Column(String(256), nullable=False)
    emergency_contact = Column(String(128), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Security/compliance fields
    email_verified = Column(Boolean, default=False)
    verification_token = Column(String(64), index=True, nullable=True)

    reset_token = Column(String(64), index=True, nullable=True)
    reset_token_expires = Column(DateTime(timezone=True), nullable=True)

    mfa_enabled = Column(Boolean, default=False)
    mfa_secret = Column(String(64), nullable=True)

    accepted_terms_at = Column(DateTime(timezone=True), nullable=True)

    failed_logins = Column(Integer, default=0)
    locked_until = Column(DateTime(timezone=True), nullable=True)
    last_login_at = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    mood_logs = relationship("MoodLog", back_populates="user")
    interactions = relationship("Interaction", back_populates="user")
    alerts = relationship("Alert", back_populates="user")


class MoodLog(Base):
    __tablename__ = "mood_logs"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    mood = Column(String(64))
    intensity = Column(Integer, default=5)
    note = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="mood_logs")


class Interaction(Base):
    __tablename__ = "interactions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    agent_type = Column(String(64))
    user_msg = Column(Text)
    agent_reply = Column(Text)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="interactions")


class Alert(Base):
    __tablename__ = "alerts"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    alert_type = Column(String(100))
    message = Column(Text)
    resolved = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", back_populates="alerts")

class Feedback(Base):
    __tablename__ = "feedback"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    rating = Column(Integer, nullable=False)
    comment = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    user = relationship("User", backref="feedback_entries")

# Add after Feedback model
import pytz

class LoginEvent(Base):
    __tablename__ = "login_events"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    user = relationship("User", backref="login_events")

def record_login(user_identifier):
    """
    Record a login event for the given username or id.
    Returns dict: {'ok': True, 'daily_logins': <int>, 'login_event_id': <int>}
    """
    db = SessionLocal()
    try:
        # Use your existing resolve_user_identifier
        uid = resolve_user_identifier(user_identifier)
        ev = LoginEvent(user_id=uid)
        db.add(ev)
        db.commit()
        db.refresh(ev)

        # Compute daily login count using IST day boundary
        ist = pytz.timezone("Asia/Kolkata")
        now_ist = datetime.now(ist)
        start_ist = now_ist.replace(hour=0, minute=0, second=0, microsecond=0)
        end_ist = start_ist + timedelta(days=1)
        # Convert to UTC to compare with DB timestamps stored in UTC
        start_utc = start_ist.astimezone(pytz.utc)
        end_utc = end_ist.astimezone(pytz.utc)

        daily_count = (
            db.query(LoginEvent)
            .filter(
                LoginEvent.user_id == uid,
                LoginEvent.created_at >= start_utc,
                LoginEvent.created_at < end_utc,
            )
            .count()
        )

        return {"ok": True, "daily_logins": daily_count, "login_event_id": ev.id}
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

def log_feedback(user_id: int, rating: int, comment: str):
    db = SessionLocal()
    try:
        fb = Feedback(user_id=user_id, rating=rating, comment=comment)
        db.add(fb)
        db.commit()
        db.refresh(fb)
        return fb
    finally:
        db.close()

def init_db():
    """Create tables if they do not exist."""
    Base.metadata.create_all(bind=engine)

# ---------- One-time additive migration for existing DB ----------
def ensure_user_columns():
    """
    One-time additive migration for existing SQLite DBs.
    Adds new columns to `users` table if they don't exist.
    Safe to call on every startup.
    """
    conn = engine.connect()
    try:
        # Get existing column names
        res = conn.exec_driver_sql("PRAGMA table_info(users);").fetchall()
        cols = {row[1] for row in res}  # row[1] = column name

        def add(col_sql: str):
            conn.exec_driver_sql(col_sql)

        # Add columns if missing (SQLite supports ADD COLUMN)
        if "email_verified" not in cols:
            add("ALTER TABLE users ADD COLUMN email_verified BOOLEAN DEFAULT 0;")
        if "verification_token" not in cols:
            add("ALTER TABLE users ADD COLUMN verification_token VARCHAR(64);")
        if "reset_token" not in cols:
            add("ALTER TABLE users ADD COLUMN reset_token VARCHAR(64);")
        if "reset_token_expires" not in cols:
            add("ALTER TABLE users ADD COLUMN reset_token_expires DATETIME;")
        if "mfa_enabled" not in cols:
            add("ALTER TABLE users ADD COLUMN mfa_enabled BOOLEAN DEFAULT 0;")
        if "mfa_secret" not in cols:
            add("ALTER TABLE users ADD COLUMN mfa_secret VARCHAR(64);")
        if "accepted_terms_at" not in cols:
            add("ALTER TABLE users ADD COLUMN accepted_terms_at DATETIME;")
        if "failed_logins" not in cols:
            add("ALTER TABLE users ADD COLUMN failed_logins INTEGER DEFAULT 0;")
        if "locked_until" not in cols:
            add("ALTER TABLE users ADD COLUMN locked_until DATETIME;")
        if "last_login_at" not in cols:
            add("ALTER TABLE users ADD COLUMN last_login_at DATETIME;")
    finally:
        conn.close()

# ---------- CRUD & Auth ----------
def create_user(username: str, password: str, email: str, emergency_contact: str):
    db = SessionLocal()
    try:
        user = User(
            username=username.strip(),
            password_hash=hash_password(password),
            email=email.strip(),
            emergency_contact=emergency_contact.strip() if emergency_contact else None,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user
    except IntegrityError:
        db.rollback()
        return None
    finally:
        db.close()

def get_user_by_username(username: str):
    db = SessionLocal()
    try:
        return db.query(User).filter(User.username == username.strip()).first()
    finally:
        db.close()


def get_user_id_by_username(username: str):
    """
    Returns the integer user id for a username, or None if not found.
    """
    u = get_user_by_username(username)
    return u.id if u else None


def resolve_user_identifier(user_identifier):
    """
    Accepts either an integer user id or a username string. Returns an integer user id
    or raises ValueError if the user cannot be resolved.
    """
    # If already an int, return it
    try:
        if isinstance(user_identifier, int):
            return user_identifier
        # If it's a string that looks like an int, convert
        if isinstance(user_identifier, str) and user_identifier.strip().isdigit():
            return int(user_identifier.strip())
    except Exception:
        pass

    # Otherwise try to resolve as username
    if isinstance(user_identifier, str):
        uid = get_user_id_by_username(user_identifier.strip())
        if uid:
            return uid

    raise ValueError(f"Could not resolve user identifier: {user_identifier}")

def authenticate_user(username: str, password: str):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == username.strip()).first()
        if not user:
            return None
        if verify_password(password, user.password_hash):
            return user
        return None
    finally:
        db.close()

def record_login_success(user_id: int):
    db = SessionLocal()
    try:
        u = db.query(User).filter(User.id == user_id).first()
        if u:
            u.failed_logins = 0
            u.locked_until = None
            u.last_login_at = _now()
            db.commit()
    finally:
        db.close()

def record_login_failure(username: str, max_attempts=5, lock_minutes=15):
    db = SessionLocal()
    try:
        u = db.query(User).filter(User.username == username.strip()).first()
        if u:
            u.failed_logins += 1
            if u.failed_logins >= max_attempts:
                u.locked_until = _now() + timedelta(minutes=lock_minutes)
            db.commit()
    finally:
        db.close()

# ---------- Email verification & password reset ----------
def set_verification_token(user_id: int):
    db = SessionLocal()
    try:
        u = db.query(User).filter(User.id == user_id).first()
        if not u:
            return None
        u.verification_token = _new_token()
        db.commit()
        db.refresh(u)
        return u.verification_token
    finally:
        db.close()

def verify_email(token: str) -> bool:
    db = SessionLocal()
    try:
        u = db.query(User).filter(User.verification_token == token).first()
        if not u:
            return False
        u.email_verified = True
        u.verification_token = None
        db.commit()
        return True
    finally:
        db.close()

def request_password_reset(email: str) -> bool:
    db = SessionLocal()
    try:
        u = db.query(User).filter(User.email == email.strip()).first()
        if not u:
            return False
        u.reset_token = _new_token()
        u.reset_token_expires = _now() + timedelta(hours=2)
        db.commit()
        return True
    finally:
        db.close()

def get_reset_token(email: str):
    db = SessionLocal()
    try:
        u = db.query(User).filter(User.email == email.strip()).first()
        if not u or not u.reset_token:
            return None
        return u.reset_token, u.reset_token_expires
    finally:
        db.close()

def reset_password(token: str, new_password: str) -> bool:
    db = SessionLocal()
    try:
        u = db.query(User).filter(User.reset_token == token).first()
        if not u or not u.reset_token_expires or u.reset_token_expires < _now():
            return False
        u.password_hash = hash_password(new_password)
        u.reset_token = None
        u.reset_token_expires = None
        db.commit()
        return True
    finally:
        db.close()

# ---------- MFA ----------
def set_mfa(user_id: int, enabled: bool, secret: str = None):
    db = SessionLocal()
    try:
        u = db.query(User).filter(User.id == user_id).first()
        if not u:
            return False
        u.mfa_enabled = enabled
        u.mfa_secret = secret
        db.commit()
        return True
    finally:
        db.close()

# ---------- App data helpers ----------
def log_interaction(user_id: int, agent_type: str, user_msg: str, agent_reply: str):
    db = SessionLocal()
    try:
        inter = Interaction(
            user_id=user_id, agent_type=agent_type, user_msg=user_msg, agent_reply=agent_reply
        )
        db.add(inter)
        db.commit()
        db.refresh(inter)
        return inter
    finally:
        db.close()


def add_mood(user_id: int, mood: str, intensity: int = 5, note: str = None):
    db = SessionLocal()
    try:
        ml = MoodLog(user_id=user_id, mood=mood, intensity=intensity, note=note)
        db.add(ml)
        db.commit()
        db.refresh(ml)
        return ml
    finally:
        db.close()

def get_mood_history(user_id: int):
    db = SessionLocal()
    try:
        rows = (
            db.query(MoodLog)
            .filter(MoodLog.user_id == user_id)
            .order_by(MoodLog.created_at.asc())
            .all()
        )
        return rows
    finally:
        db.close()

def create_alert(user_id: int, alert_type: str, message: str):
    db = SessionLocal()
    try:
        a = Alert(user_id=user_id, alert_type=alert_type, message=message)
        db.add(a)
        db.commit()
        db.refresh(a)
        return a
    finally:
        db.close()

def delete_user_interactions(user_id: int):
    """Deletes all interactions for a given user, excluding feedback."""
    db = SessionLocal()
    try:
        # This deletes all rows in the 'interactions' table that match the user_id
        # and are not feedback entries.
        db.query(Interaction).filter(
            Interaction.user_id == user_id,
            Interaction.agent_type != "feedback"
        ).delete(synchronize_session=False)
        db.commit()
        return True
    except Exception as e:
        print(f"Error deleting interactions for user {user_id}: {e}")
        db.rollback()
        return False
    finally:
        db.close()

def get_user_metrics(user_id: int) -> dict:
    """
    Aggregates and calculates key metrics for a specific user.
    Now returns 'daily_logins' and 'conversation_streak' (consecutive-day streak based on login events, IST).
    """
    db = SessionLocal()
    try:
        # 1. Interaction Metrics (unchanged)
        interactions = db.query(Interaction).filter(Interaction.user_id == user_id).all()
        total_interactions = len(interactions)

        agent_usage = db.query(Interaction.agent_type, func.count(Interaction.agent_type)).\
            filter(Interaction.user_id == user_id, Interaction.agent_type != 'feedback').\
            group_by(Interaction.agent_type).all()
        agent_usage_dict = {agent: count for agent, count in agent_usage}

        # 2. Login-based metrics (daily_logins and consecutive-day streak)
        try:
            login_rows = db.query(LoginEvent.created_at).filter(LoginEvent.user_id == user_id).all()
            ist = pytz.timezone("Asia/Kolkata")
            today_ist = datetime.now(ist).date()

            # Count ALL login events for today (not just unique dates)
            daily_logins = 0
            login_dates_set = set()
            for (ts,) in login_rows:
                if ts is None:
                    continue
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                dt_ist = ts.astimezone(ist)
                if dt_ist.date() == today_ist:
                    daily_logins += 1
                login_dates_set.add(dt_ist.date())

            # conversation_streak: consecutive-day streak (based on unique login dates)
            login_dates = sorted(list(login_dates_set), reverse=True)
            streak = 0
            if login_dates:
                current = today_ist if today_ist in login_dates else login_dates[0]
                while current in login_dates:
                    streak += 1
                    current = current - timedelta(days=1)
        except Exception:
            login_rows = []
            daily_logins = 0
            streak = 0

        # 3. Feedback Metrics (unchanged)
        avg_feedback = db.query(func.avg(Feedback.rating)).filter(Feedback.user_id == user_id).scalar() or 0

        # 4. Mood Log Metrics
        total_moods_logged = db.query(MoodLog).filter(MoodLog.user_id == user_id).count()

        return {
            "total_interactions": total_interactions,
            "agent_usage": agent_usage_dict,
            "conversation_streak": streak,
            "daily_logins": daily_logins,
            "avg_feedback_rating": round(avg_feedback, 2),
            "total_moods_logged": total_moods_logged,
        }
    finally:
        db.close()

if __name__ == "__main__":
    init_db()
    # Optional: run migration helper if invoking directly
    ensure_user_columns()
    print("DB initialized & columns ensured.")
