import cv2
import streamlit as st
import sys
import bcrypt
import string
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
from pymongo import MongoClient
from datetime import datetime, timezone
from tzlocal import get_localzone
import pytz
import pandas as pd
import numpy as np
import io
import base64
import re
import random
import uuid


# =============================================
# PAGE CONFIGURATION (MUST BE FIRST STREAMLIT COMMAND)
# =============================================
st.set_page_config(
    page_title="Object Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)

from streamlit_cookies_manager import EncryptedCookieManager

# =============================================
# ---- CONFIGURATION -----
# =============================================
# MongoDB Configuration
MONGO_URI = "mongodb://localhost:27017"  # or replace with your Atlas URI
DB_NAME = "object_detection"
COLLECTION_NAME = "detections"

# Connect to MongoDB
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]
USERS_COLLECTION = "users"
users_collection = db[USERS_COLLECTION]


# Get the absolute path of the current file
FILE = Path(__file__).resolve()

# Get the parent directory of the current file
ROOT = FILE.parent

# Add the root path to the sys.path list
if ROOT not in sys.path:
    sys.path.append(str(ROOT))

# Get the relative path of the root directory with respect to the current working directory
ROOT = ROOT.relative_to(Path.cwd())

# Sources
IMAGE = 'Image'

# Image Config
IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = IMAGES_DIR / 'image2.jpg'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'detectedimage2.jpg'

# Model Configurations
MODEL_DIR = ROOT / 'weights'
DETECTION_MODEL = MODEL_DIR / 'yolo11n.pt'
SEGMENTATION_MODEL = MODEL_DIR / 'yolo11n-seg.pt'


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

def reset_password(email):
    user = users_collection.find_one({"email": email})
    if not user:
        return False, "Email not found."
    
    new_pass = generate_temp_password()
    hashed = hash_password(new_pass)
    users_collection.update_one({"email": email}, {"$set": {"password": hashed}})
    
    print(f"[DEBUG] New temporary password for {email}: {new_pass}")
    return True, f"A new temporary password has been generated and sent to your email."

def generate_temp_password(length=10):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


# =============================================
# ——— AUTH UI ———
# =============================================
# Session management
cookies = EncryptedCookieManager(password="super-secret-key")
if not cookies.ready():
    st.stop()

# Persistent session check using cookies
if "authenticated" not in st.session_state:
    user_id = cookies.get("user_id")
    username = cookies.get("username")
    if user_id and username:
        st.session_state.authenticated = True
        st.session_state.username = username
        st.session_state.user_id = user_id
    else:
        st.session_state.authenticated = False
        st.session_state.username = None

def auth_page():
    st.title("User Authentication")

    tab1, tab2, tab3, tab4 = st.tabs(["Login", "Register", "Forgot Password", "Forgot Username"])

    # --- Login ---
    with tab1:
        st.subheader("Login")
        login_username = st.text_input("Username", key="login_user")
        login_password = st.text_input("Password", type="password", key="login_pass")
        if st.button("Login"):
            user = login_user(login_username, login_password)
            if user:
                st.session_state.authenticated = True
                st.session_state.username = user["username"]
                st.session_state.user_id = user["user_id"]
                cookies["user_id"] = user["user_id"]
                cookies["username"] = user["username"]
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid username or password.")

    # --- Register ---
    with tab2:
        st.subheader("Register")
        reg_username = st.text_input("Username", key="reg_user")
        reg_email = st.text_input("Email", key="reg_email")
        reg_password = st.text_input("Password", type="password", key="reg_pass")
        reg_confirm_password = st.text_input("Confirm Password", type="password", key="reg_confirm_pass")
        if st.button("Register"):
            status, msg = register_user(reg_username, reg_email, reg_password, reg_confirm_password)
            if status:
                st.success("Register Succsessful" + msg)
            else:
                st.error("Register Failed" + msg)

    # --- Forgot Password ---
    with tab3:
        st.subheader("Forgot Password?")
        reset_email = st.text_input("Enter your registered email", key="reset_email")
        if st.button("Reset Password"):
            success, msg = reset_password(reset_email)
            if success:
                st.success("Password has been sent" + msg)
                st.info("Temporary password printed in console for dev/testing.")
            else:
                st.error("Error" + msg)

    # --- Forgot Username ---
    with tab4:
        st.subheader("Forgot Username?")
        lookup_email = st.text_input("Enter your email", key="lookup_email")
        if st.button("Find Username"):
            username = get_username_by_email(lookup_email)
            if username:
                st.success(f"Your username is: `{username}`")
            else:
                st.error("No user found with this email.")

if not st.session_state.authenticated:
    auth_page()
    st.stop()

# --- Logout ---
with st.sidebar:
    st.write(f"Logged in as: `{st.session_state.username}`")
    if st.button("Logout"):
        del cookies["user_id"]
        del cookies["username"]
        st.session_state.authenticated = False
        st.session_state.username = None
        st.session_state.user_id = None
        st.rerun()


# =============================================
# ------- UI ------
# =============================================

# CSS for styling
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .sidebar .sidebar-content {
            background-color: #343a40;
            color: white;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            padding: 10px 24px;
            font-weight: bold;
            width: 100%;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .stSelectbox, .stSlider {
            margin-bottom: 20px;
        }
        .stImage {
            border-radius: 10px;
            box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
        }
        .stDataFrame {
            border-radius: 10px;
            box-shadow: 0 4px 8px 0 rgba(0,0,0,0.1);
        }
        .header-text {
            font-size: 2.5rem;
            font-weight: 700;
            color: #2c3e50;
            margin-bottom: 1rem;
        }
        .subheader-text {
            font-size: 1.2rem;
            color: #7f8c8d;
            margin-bottom: 2rem;
        }
        .metric-card {
            background: white;
            border-radius: 10px;
            padding: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .metric-title {
            font-size: 0.9rem;
            color: #7f8c8d;
            margin-bottom: 5px;
        }
        .metric-value {
            font-size: 1.5rem;
            font-weight: bold;
            color: #2c3e50;
        }
        .detection-button-container {
            margin-top: 2rem;
            text-align: center;
        }
        .paste-container {
            border: 2px dashed #ccc;
            border-radius: 5px;
            padding: 20px;
            text-align: center;
            margin-bottom: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Header with title 
st.markdown('<p class="header-text">System for Object Recognition</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader-text">Real-time object detection and segmentation</p>', unsafe_allow_html=True)

# SideBar
with st.sidebar:
    st.markdown("""
        <style>
            .sidebar .sidebar-content {
                background-image: linear-gradient(#343a40,#2c3e50);
                color: white;
            }
            .sidebar .stRadio label {
                color: white;
            }
            .sidebar .stSlider label {
                color: white;
            }
            .sidebar .stFileUploader label {
                color: white;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # =============================================
    # ---- MODEL CONFIGURATION ----
    # =============================================
    st.header("Model Configuration")
    
    # Choose Model: Detection or Segmentation 
    model_type = st.radio(
        "Select Task Type:",
        ["Detection", "Segmentation",],
        index=0,
        help="Choose between object detection and instance segmentation"
    )
    
    st.markdown("---")
    
    # Select Confidence Value
    confidence_value = st.slider(
        "Confidence Threshold", 
        min_value=0, 
        max_value=100, 
        value=40,
        help="Adjust the minimum confidence level for detections"
    )
    confidence_value = float(confidence_value) / 100
    
    # Visual indicator for confidence level
    st.markdown(f"""
        <div style="background: linear-gradient(90deg, #e74c3c {confidence_value*100}%, #ecf0f1 {confidence_value*100}%);
                    height: 8px; 
                    border-radius: 4px;
                    margin-bottom: 20px;"></div>
    """, unsafe_allow_html=True)
    
    # Class Selection
    CLASSES = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
        "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
        "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]
    
    # Image Source Selection
    st.subheader("Image Source")
    image_source = st.radio(
        "Select image source:",
        ["Upload an image", "Paste from clipboard"],
        index=0,
        help="Choose how to provide the image for detection"
    )
    
    # Image Upload/Paste Section
    st.subheader("Image Configuration")
    source_image = None
    
    if image_source == "Upload an image":
        source_image = st.file_uploader(
            "Upload an image",
            type=("jpg", "png", "jpeg", "bmp", "webp"),
            help="Upload an image for object detection",
            key="file_uploader"
        )
    else:
        paste_data = st.text_area("Paste image here (as base64 or URL)", "", height=100, key="paste_area")
        
        if paste_data:
            try:
                if paste_data.startswith("data:image"):
                    header, encoded = paste_data.split(",", 1)
                    image_data = base64.b64decode(encoded)
                    source_image = io.BytesIO(image_data)
                elif paste_data.startswith(("http://", "https://")):
                    import requests
                    from io import BytesIO
                    response = requests.get(paste_data)
                    source_image = BytesIO(response.content)
                else:
                    image_data = base64.b64decode(paste_data)
                    source_image = io.BytesIO(image_data)
            except:
                st.error("Could not process the pasted image. Please try another method.")



# Selecting Detection or Segmentation Model
if model_type == 'Detection':
    model_path = Path(DETECTION_MODEL)
elif model_type == 'Segmentation':
    model_path = Path(SEGMENTATION_MODEL)

# Load the YOLO Model
try:
    model = YOLO(model_path)
except Exception as e:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(e)

# =============================================
# Main PAGE (FRONT PAGE)
# =============================================

tab1, tab2, tab3 = st.tabs(["Image Detection", "Statistics", "Detection History"])

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Input Image")
        try:
            if source_image is None:
                default_image_path = str(DEFAULT_IMAGE)
                default_image = Image.open(default_image_path)
                st.image(default_image_path, 
                         caption="Default Image - Upload or paste your own image to see detection results", 
                         use_container_width=True)
            else:
                if isinstance(source_image, io.BytesIO):
                    # Reset pointer to start if it's a BytesIO object
                    source_image.seek(0)
                uploaded_image = Image.open(source_image)
                st.image(uploaded_image, 
                         caption="Input Image - Click 'Detect Objects' to process", 
                         use_container_width=True)
        except Exception as e:
            st.error("Error Occurred While Opening the Image")
            st.error(e)
    
    with col2:
        st.subheader("Detection Results")
        try:
            if source_image is None:
                default_detected_image_path = str(DEFAULT_DETECT_IMAGE)
                default_detected_image = Image.open(default_detected_image_path)
                st.image(default_detected_image_path, 
                         caption="Sample Detection - Upload or paste your own image to see live results", 
                         use_container_width=True)
        except Exception as e:
            st.error("Error Occurred While Processing the Image")
            st.error(e)
    st.markdown("---")

# =============================================
# ---- DETECTION BUTTON -----
# =============================================

if source_image is not None:
    st.markdown('<div class="detection-button-container">', unsafe_allow_html=True)
    if st.button(f"Detect Objects ({model_type})", key="detect_button"):
        with st.spinner(f"Processing {model_type}..."):
            try:
                if isinstance(source_image, io.BytesIO):
                    source_image.seek(0)
                uploaded_image = Image.open(source_image)
                
                # Run detection
                result = model.predict(uploaded_image, conf=confidence_value)
                boxes = result[0].boxes
                result_plotted = result[0].plot()[:, :, ::-1]
                
                # Display results in the right column
                with tab1:
                    with col2:
                        st.image(result_plotted, 
                                 caption=f"{model_type} Results (Confidence: {confidence_value*100}%)", 
                                 use_container_width=True)
                
                # Object Counting
                class_counts = {}
                for box in boxes:
                    class_id = int(box.cls)
                    class_name = CLASSES[class_id]
                    if class_name in class_counts:
                        class_counts[class_name] += 1
                    else:
                        class_counts[class_name] = 1
                

                # Get image size (width, height)
                image_width, image_height = uploaded_image.size

                # Extract YOLO model version or filename
                model_name = getattr(model, 'model', None)
                model_version = str(model_name) if model_name else str(model_path.name)

                # Preparing detection data
                detection_data = {
                    "timestamp": datetime.now(timezone.utc),
                    "model_type": model_type,
                    "confidence_threshold": confidence_value,
                    "source": image_source,
                    "image_name": source_image.name if image_source == "Upload an image" and hasattr(source_image, 'name') else "pasted_image",
                    "image_resolution": {
                        "width": image_width,
                        "height": image_height
                    },
                    "object_counts": class_counts,
                    "objects": [],
                    "user_id": st.session_state.user_id
                }

                # Add each detected object
                for i, box in enumerate(boxes):
                    detection_data["objects"].append({
                        "object_id": i,
                        "class": CLASSES[int(box.cls)],
                        "confidence": float(box.conf),
                        "box_xywh": box.xywh.tolist()[0]
                    })

                image_pil = Image.fromarray(result_plotted)
                buffered = io.BytesIO()
                image_pil.save(buffered, format="JPEG")
                detection_data["image_bytes"] = base64.b64encode(buffered.getvalue()).decode("utf-8")

                # Determine unique classes detected
                unique_classes = list(class_counts.keys())

                # Choose target collection based on classes
                if len(unique_classes) == 1:
                    target_collection_name = unique_classes[0].replace(" ", "_")  # e.g., "traffic light" → "traffic_light"
                else:
                    target_collection_name = "Multiclass_objects"

                # Insert into appropriate collection
                try:
                    target_collection = db[target_collection_name]
                    target_collection.insert_one(detection_data)
                    st.success(f"Detection saved to MongoDB collection: '{target_collection_name}'")
                except Exception as e:
                    st.error("Failed to store detection in MongoDB:")
                    st.error(e)

                # Store results in session state for the statistics tab
                st.session_state.detection_results = {
                    "image": result_plotted,
                    "counts": class_counts,
                    "boxes": boxes,
                    "model_type": model_type
                }
            except Exception as e:
                st.error("Error during detection:")
                st.error(e)
    st.markdown('</div>', unsafe_allow_html=True)


with tab2:
    if 'detection_results' not in st.session_state:
        st.warning("Run a detection first to see statistics")
    else:
        results = st.session_state.detection_results
        st.subheader(f"{results['model_type']} Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="metric-card"><div class="metric-title">Total Objects Detected</div>'
                        f'<div class="metric-value">{sum(results["counts"].values())}</div></div>', 
                        unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card"><div class="metric-title">Unique Classes</div>'
                        f'<div class="metric-value">{len(results["counts"])}</div></div>', 
                        unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card"><div class="metric-title">Model Confidence</div>'
                        f'<div class="metric-value">{confidence_value*100:.0f}%</div></div>', 
                        unsafe_allow_html=True)
        
        # ---- Statistics display Object Counts in a Table with sorting ----
        st.subheader("Object Counts")
        count_df = pd.DataFrame(list(results["counts"].items()), columns=["Class", "Count"])
        st.dataframe(
            count_df.sort_values("Count", ascending=False),
            use_container_width=True,
            height=min(400, 50 + 35 * len(count_df))
        )
        # Show detection details in an expander
        with st.expander("Detailed Detection Data"):
            st.write("Raw detection data from the model:")
            for i, box in enumerate(results["boxes"]):
                st.json({
                    "object_id": i,
                    "class": CLASSES[int(box.cls)],
                    "confidence": float(box.conf),
                    "coordinates": box.xywh.tolist()
                })

with tab3:
    st.subheader("Detection History")
    local_tz = get_localzone()
    import history

    user_id = st.session_state.user_id
    class_filter = st.text_input("Filter by class name (optional)")
    source_filter = st.selectbox("Filter by source (optional)", ["", "Upload an image", "Paste from clipboard"])
    
    history_docs = history.get_detection_history(
        user_id=user_id,
        class_filter=class_filter if class_filter else None,
        source_filter=source_filter if source_filter else None,
        limit=100
    )

    history_docs = sorted(
        history_docs,
        key=lambda doc: doc.get("timestamp", datetime.min.replace(tzinfo=timezone.utc)),
        reverse=True
    )

    if not history_docs:
        st.info("No detection history found.")
    else:
        for i, doc in enumerate(history_docs):
            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown(f"**{i+1}. {doc.get('image_name', 'Unnamed Image')}**")
                image_data = doc.get("image_bytes")
                image_data = doc.get("image_bytes")
                if image_data:
                    try:
                        decoded_image = base64.b64decode(image_data)
                        image = Image.open(io.BytesIO(decoded_image))
                        st.image(image, caption="Detected Image", use_container_width =True)
                    except Exception as e:
                        st.warning(f"Failed to decode or display image: {e}")


            with col2:
                st.markdown("#### Information:")
                utc_time = doc.get("timestamp")
                if utc_time.tzinfo is None:
                    utc_time = utc_time.replace(tzinfo=pytz.UTC)
                local_time = utc_time.astimezone(local_tz)
                local_time_str = local_time.strftime("%Y-%m-%d %H:%M:%S %Z")
                st.markdown(f"- **Date:** {local_time_str}")
                st.markdown(f"- **Model:** {doc.get('model_type')}")
                st.markdown(f"- **Source:** {doc.get('source')}")
                confidence_threshold = doc.get("confidence_threshold", "N/A")
                st.write("- **Confidence Threshold:**", confidence_threshold)
                st.markdown("#### Detected Classes:")
                for cls, count in doc.get("object_counts", {}).items():
                    st.markdown(f"- {cls}: {count}")

            st.markdown("---")


