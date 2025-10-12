# security.py
import pyotp, qrcode
from io import BytesIO
import base64

def new_totp_secret():
    return pyotp.random_base32()

def verify_totp(secret: str, code: str) -> bool:
    if not secret or not code: return False
    totp = pyotp.TOTP(secret)
    # valid_window allows minor clock skews
    return bool(totp.verify(code, valid_window=1))

def totp_provisioning_uri(secret: str, username: str, issuer_name: str = "Synermind"):
    return pyotp.TOTP(secret).provisioning_uri(name=username, issuer_name=issuer_name)

def qr_png_data_uri(content: str) -> str:
    img = qrcode.make(content)
    buf = BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{b64}"
