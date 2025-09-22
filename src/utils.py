import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import streamlit as st
import bcrypt
import string
import re
import random
import uuid
from mongo import users_collection


# =============================================
# LOGIN/REGISTRATION UI
# =============================================


EMAIL_REGEX = r"^[\w\.-]+@[\w\.-]+\.\w+$"
PASSWORD_REGEX = r"^(?=.*[a-z])(?=.*[A-Z])(?=.*\d).{6,}$"

def is_valid_email(email):
    return re.match(EMAIL_REGEX, email)

def is_strong_password(password):
    return re.match(PASSWORD_REGEX, password)

def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def check_password(password, hashed):
    return bcrypt.checkpw(password.encode('utf-8'), hashed)

def register_user(username, email, password, confirm_password):
    username = username.lower().strip()  # convert to lowercase

    if password != confirm_password:
        return False, "Passwords do not match."
    
    # Case-insensitive check is no longer needed if we store everything lowercase
    if users_collection.find_one({"username": username}):
        return False, "Username already exists."

    if users_collection.find_one({"email": email}):
        return False, "Email already registered."
    if not is_valid_email(email):
        return False, "Invalid email format."
    if not is_strong_password(password):
        return False, "Password must contain at least one uppercase letter, one lowercase letter, one number, and be at least 6 characters long."

    hashed = hash_password(password)
    user_id = str(uuid.uuid4()) 

    users_collection.insert_one({
        "user_id": user_id,
        "username": username,
        "email": email,
        "password": hashed
    })
    return True, "User registered successfully."

def login_user(username, password):
    username = username.lower().strip()  # convert to lowercase
    user = users_collection.find_one({"username": username})
    if user and check_password(password, user['password']):
        return user
    return None

def get_username_by_email(email):
    user = users_collection.find_one({"email": email})
    if user:
        return user["username"]
    return None

def send_username(email):
    user = users_collection.find_one({"email": email})
    if not user:
        return False, "Email not found."

    username = user["username"]

    subject = "Username Recovery - Object Detection System"
    body = f"""Hello,

    Your registered username is: {username}

    If you didn’t request this, please ignore this email.
    """

    email_sent = send_email(email, subject, body)

    if email_sent:
        return True, "Your username has been sent to your email."
    else:
        # fallback for dev visibility
        print(f"[DEBUG] Username for {email}: {username}")
        return False, "We couldn't send the email. Your username is printed in the server console."

def reset_password(email):
    user = users_collection.find_one({"email": email})
    if not user:
        return False, "Email not found."
    
    new_pass = generate_temp_password()
    hashed = hash_password(new_pass)
    users_collection.update_one({"email": email}, {"$set": {"password": hashed}})
    
    subject = "Password Reset - Object Detection System"
    body = f"""Hello {user['username']},

    Your temporary password is: {new_pass}

    Log in with this password and change it immediately.
    """

    email_sent = send_email(email, subject, body)

    if email_sent:
        return True, "A new temporary password has been sent to your email."
    else:
        # Fallback for dev visibility
        print(f"[DEBUG] Temporary password for {email}: {new_pass}")
        return False, "We couldn't send the email. Use the temporary password printed in the server console."
    

def generate_temp_password(length=10):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

# =============================================
# EMAIL CONFIGURATION
# =============================================
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "objectdetectionsystem1@gmail.com"
SENDER_PASSWORD = "tzvgqobgzseiyyng"  #AppPassword
def send_email(recipient_email, subject, body):
    try:
        msg = MIMEMultipart()
        msg["From"] = SENDER_EMAIL
        msg["To"] = recipient_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)

        return True
    except Exception as e:
        error_msg = f"[EMAIL ERROR] {type(e).__name__}: {str(e)}"
        print(error_msg)   
        st.error(error_msg)  
        return False