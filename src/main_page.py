import cv2
import sys
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
from datetime import datetime, timezone
from tzlocal import get_localzone
from st_img_pastebutton import paste
import pytz
import pandas as pd
import numpy as np
import io
import tensorflow as tf
import base64
import streamlit as st
from utils import check_password, is_valid_email
from mongo import users_collection,db

# =============================================
# ---- CONFIGURATION -----
# =============================================



# Get the absolute path of the current file
FILE = Path(__file__).resolve()

# Get the parent directory of the current file
ROOT = FILE.parent

# Add the root path to the sys.path list
if ROOT not in sys.path:
    sys.path.append(str(ROOT))

# Get the relative path of the root directory with respect to the current working directory
ROOT = ROOT.relative_to(Path.cwd())


# Image Config
IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = IMAGES_DIR / 'image2.jpg'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'detectedimage2.jpg'

# Model Configurations
MODEL_DIR = ROOT / 'weights'
DETECTION_MODEL = MODEL_DIR / 'yolo11n.pt'
SEGMENTATION_MODEL = MODEL_DIR / 'yolo11n-seg.pt'

CUSTOM_YOLO_MODEL = MODEL_DIR / 'my_model.pt'

CUSTOM_MODEL_PATH = MODEL_DIR / 'simple_object_detection.h5'

@st.cache_resource
def load_yolo_model(path):
    return YOLO(path)

@st.cache_resource
def load_tf_model(path):
    return tf.keras.models.load_model(path, compile=False)

# preload models
yolo_det_model = load_yolo_model(DETECTION_MODEL)
yolo_seg_model = load_yolo_model(SEGMENTATION_MODEL)
yolo_custom_model = load_yolo_model(CUSTOM_YOLO_MODEL)

try:
    tf_custom_model = load_tf_model(CUSTOM_MODEL_PATH)
except Exception as e:
    tf_custom_model = None



def main_page():
    
        
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

    if "page" not in st.session_state:
        st.session_state.page = "home"

    if "account_action" not in st.session_state:
        st.session_state.account_action = "Home"

    elif st.session_state.page == "change_email":
        st.title("Change Email")
        password_for_email = st.text_input("Password", type="password")
        new_email = st.text_input("New Email")

        if st.button("Change Email"):
            user = users_collection.find_one({"user_id": st.session_state.user_id})
            if user and check_password(password_for_email, user['password']):
                if is_valid_email(new_email) and not users_collection.find_one({"email": new_email}):
                    users_collection.update_one(
                        {"user_id": user["user_id"]},
                        {"$set": {"email": new_email}}
                    )
                    st.success("Email updated successfully.")
                else:
                    st.error("Invalid or already registered email.")
            else:
                st.error("Password is incorrect.")


        st.stop()   # prevent main page from rendering
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
            ["YOLO General Detection", "YOLO General Segmentation","YOLO Model (Specific)", "TensorFlow Model (Specific)"],
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
            ["Upload an image", "Paste from clipboard (text input)", "Paste from clipboard (button)"],
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

        elif image_source == "Paste from clipboard (text input)":
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

        elif image_source == "Paste from clipboard (button)":
            image_data = paste(label="Paste from Clipboard", key="image_clipboard")
            if image_data is not None:
                try:
                    header, encoded = image_data.split(",", 1)
                    binary_data = base64.b64decode(encoded)
                    source_image = io.BytesIO(binary_data)
                    st.image(source_image, caption="Pasted from Clipboard", use_container_width=True)
                except Exception as e:
                    st.error(f"Failed to process clipboard image: {e}")




    # Selecting Detection or Segmentation Model
    if model_type == 'YOLO General Detection':
        model = yolo_det_model
        class_map = model.names
    elif model_type == 'YOLO General Segmentation':
        model = yolo_seg_model
        class_map = model.names
    elif model_type == 'YOLO Model (Specific)':
        model = yolo_custom_model
        class_map = model.names
    elif model_type == 'TensorFlow Model (Specific)':
        model = tf_custom_model
        class_map = None



    tab1, tab2, tab3, tab4 = st.tabs(["Image Detection", "Statistics", "Detection History", "Overall Statistics"])


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
    # ---- IMAGE RESIZE CONFIGURATION TOGGLE ----
    # =============================================

    def update_width_from_slider():
        st.session_state.resize_width = st.session_state.width_slider
        st.session_state.width_input = st.session_state.width_slider

    def update_width_from_input():
        st.session_state.resize_width = st.session_state.width_input
        st.session_state.width_slider = st.session_state.width_input

    def update_height_from_slider():
        st.session_state.resize_height = st.session_state.height_slider
        st.session_state.height_input = st.session_state.height_slider

    def update_height_from_input():
        st.session_state.resize_height = st.session_state.height_input
        st.session_state.height_slider = st.session_state.height_input


    # Sidebar toggle
    resize_mode = st.sidebar.radio(
        "Image Resize Mode",
        ("Disabled", "Enabled"),
        index=0
    )

    if source_image is not None and resize_mode == "Enabled":
        st.subheader("Image Resize")

        # Get original image size
        try:
            if isinstance(source_image, io.BytesIO):
                source_image.seek(0)
            temp_img = Image.open(source_image)
            orig_width, orig_height = temp_img.size
        except Exception:
            orig_width, orig_height = 640, 480  # fallback

        st.info(f"Original Size: {orig_width} x {orig_height} px")

        # Initialize session state
        if "resize_width" not in st.session_state:
            st.session_state.resize_width = orig_width
        if "resize_height" not in st.session_state:
            st.session_state.resize_height = orig_height

        # Checkbox for aspect ratio
        keep_aspect = st.checkbox("Keep Aspect Ratio", value=True)

        aspect_ratio = orig_width / orig_height

        if keep_aspect:
            # --- Single width slider (separate key) ---
            new_width = st.slider(
                "Width (px)",
                min_value=50,
                max_value=5000,
                value=st.session_state.get("resize_width", orig_width),
                step=1,
                key="aspect_width_slider"
            )

            # Calculate height on-the-fly
            new_height = int(new_width / aspect_ratio)

            # Force update session state immediately
            st.session_state.resize_width = new_width
            st.session_state.resize_height = new_height

            st.write(f"Auto-scaled Height: {new_height}px")


        else:
            # --- Independent width/height sliders ---
            col_w1, col_w2 = st.columns([3, 1])
            with col_w1:
                st.slider(
                    "Width (px)",
                    min_value=50,
                    max_value=5000,
                    value=st.session_state.resize_width,
                    step=1,
                    key="width_slider",
                    on_change=update_width_from_slider
                )
            with col_w2:
                st.number_input(
                    "Width",
                    min_value=50,
                    max_value=5000,
                    value=st.session_state.resize_width,
                    step=1,
                    key="width_input",
                    on_change=update_width_from_input
                )

            col_h1, col_h2 = st.columns([3, 1])
            with col_h1:
                st.slider(
                    "Height (px)",
                    min_value=50,
                    max_value=5000,
                    value=st.session_state.resize_height,
                    step=1,
                    key="height_slider",
                    on_change=update_height_from_slider
                )
            with col_h2:
                st.number_input(
                    "Height",
                    min_value=50,
                    max_value=5000,
                    value=st.session_state.resize_height,
                    step=1,
                    key="height_input",
                    on_change=update_height_from_input
                )

        # --- Apply resize ---
        try:
            if isinstance(source_image, io.BytesIO):
                source_image.seek(0)
            img_pil = Image.open(source_image)
            img_np = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

            resized_img = cv2.resize(
                img_np,
                (st.session_state.resize_width, st.session_state.resize_height),
                interpolation=cv2.INTER_AREA
            )

            resized_pil = Image.fromarray(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB))
            buf = io.BytesIO()
            resized_pil.save(buf, format="JPEG")
            buf.seek(0)
            source_image = buf  # update source image

            st.image(
                resized_pil,
                caption=f"Resized Image ({st.session_state.resize_width}x{st.session_state.resize_height})",
                use_container_width=True
            )

        except Exception as e:
            st.error(f"Error resizing image: {e}")


    # =============================================
    # ---- DETECTION BUTTON -----
    # =============================================

    if source_image is not None:
        st.markdown('<div class="detection-button-container">', unsafe_allow_html=True)
        if st.button(f"Detect Objects ({model_type})", key="detect_button"):
            with st.spinner(f"Processing {model_type}..."):
                try:
                    uploaded_image = Image.open(source_image)

                    # ---------------- YOLO (Detection / Segmentation / Custom YOLO) ----------------
                    if model_type in ["YOLO General Detection", "YOLO General Segmentation", "YOLO Model (Specific)"]:
                        result = model.predict(uploaded_image, conf=confidence_value)
                        boxes = result[0].boxes
                        result_plotted = result[0].plot()[:, :, ::-1]

                        with tab1:
                            with col2:
                                st.image(result_plotted,
                                        caption=f"{model_type} Results (Confidence: {confidence_value*100:.1f}%)",
                                        use_container_width=True)

                        # броене на обекти
                        class_counts = {}
                        for box in boxes:
                            class_id = int(box.cls)
                            class_name = class_map[class_id]
                            class_counts[class_name] = class_counts.get(class_name, 0) + 1

                        # размери на изображението
                        image_width, image_height = uploaded_image.size

                        # подготовка на данни за MongoDB
                        detection_data = {
                            "timestamp": datetime.now(timezone.utc),
                            "model_type": model_type,
                            "confidence_threshold": confidence_value,
                            "source": image_source,
                            "image_name": source_image.name if image_source == "Upload an image" and hasattr(source_image, 'name') else "pasted_image",
                            "image_resolution": {"width": image_width, "height": image_height},
                            "object_counts": class_counts,
                            "objects": [],
                            "user_id": st.session_state.user_id
                        }

                        # добавяне на всеки обект
                        for i, box in enumerate(boxes):
                            detection_data["objects"].append({
                                "object_id": i,
                                "class": class_map[int(box.cls)],  # <-- use model.names
                                "confidence": float(box.conf.item() if hasattr(box.conf, "item") else box.conf),
                                "box_xywh": box.xywh.tolist()[0]
                            })

                        # записване изображение като base64
                        image_pil = Image.fromarray(result_plotted)
                        buffered = io.BytesIO()
                        image_pil.save(buffered, format="JPEG")
                        detection_data["image_bytes"] = base64.b64encode(buffered.getvalue()).decode("utf-8")

                        # избор на колекция
                        unique_classes = list(class_counts.keys())
                        if len(unique_classes) == 0:
                            target_collection_name = "no_detections"
                        elif len(unique_classes) == 1:
                            target_collection_name = unique_classes[0].replace(" ", "_")
                        else:
                            target_collection_name = "Multiclass_objects"

                        try:
                            db[target_collection_name].insert_one(detection_data)
                            st.success(f"Detection saved to database collection")
                        except Exception as e:
                            st.error("Failed to store detection in MongoDB:")
                            st.error(e)

                        # запазване в session_state
                        st.session_state.detection_results = {
                            "image": result_plotted,
                            "counts": class_counts,
                            "boxes": boxes,
                            "model_type": model_type,
                            "class_map": class_map
                        }

                    # ---------------- Custom TensorFlow Model ----------------
                    elif model_type == "TensorFlow Model (Specific)":
                        if model is None:
                            st.error("Custom model not loaded.")
                        else:
                            # Преобработка
                            img = uploaded_image.convert("L")
                            img = img.resize((340, 340))
                            img_arr = np.array(img) / 255.0
                            img_arr = np.expand_dims(img_arr, axis=(0, -1))

                            # Предсказване
                            pred_bbox, pred_class_logits = model.predict(img_arr)
                            pred_bbox = np.squeeze(pred_bbox)
                            pred_class = np.argmax(pred_class_logits, axis=-1)[0]

                            # Преоразмеряване обратно към оригинала
                            w, h = uploaded_image.size
                            xmin, ymin, xmax, ymax = (
                                int(pred_bbox[0] * w),
                                int(pred_bbox[1] * h),
                                int(pred_bbox[2] * w),
                                int(pred_bbox[3] * h),
                            )

                            # Рисуване
                            img_np = np.array(uploaded_image)
                            cv2.rectangle(img_np, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                            label = "Water Gun"
                            cv2.putText(img_np, label, (xmin, max(ymin - 10, 0)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                            with tab1:
                                with col2:
                                    st.image(img_np, caption="Custom Model Detection", use_container_width=True)

                            # Подготовка на данни за MongoDB
                            detection_data = {
                                "timestamp": datetime.now(timezone.utc),
                                "model_type": model_type,
                                "confidence_threshold": confidence_value,
                                "source": image_source,
                                "image_name": source_image.name if hasattr(source_image, 'name') else "pasted_image",
                                "image_resolution": {"width": w, "height": h},
                                "object_counts": {"Water Gun": 1},
                                "objects": [{
                                    "object_id": 0,
                                    "class": "Water Gun",
                                    "confidence": float(np.max(pred_class_logits)),
                                    "box_xyxy": [xmin, ymin, xmax, ymax]
                                }],
                                "user_id": st.session_state.user_id
                            }

                            buffered = io.BytesIO()
                            Image.fromarray(img_np).save(buffered, format="JPEG")
                            detection_data["image_bytes"] = base64.b64encode(buffered.getvalue()).decode("utf-8")

                            try:
                                db["Water_Gun"].insert_one(detection_data)
                                st.success("Detection saved to MongoDB collection: 'Water_Gun'")
                            except Exception as e:
                                st.error("Failed to store detection in MongoDB:")
                                st.error(e)

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
                            f'<div class="metric-value">{sum(results.get("counts", {}).values())}</div></div>', 
                            unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card"><div class="metric-title">Unique Classes</div>'
                            f'<div class="metric-value">{len(results.get("counts", {}))}</div></div>', 
                            unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card"><div class="metric-title">Model Confidence</div>'
                            f'<div class="metric-value">{confidence_value*100:.0f}%</div></div>', 
                            unsafe_allow_html=True)
            
            # ---- Statistics display Object Counts in a Table with sorting ----
            st.subheader("Object Counts")

            # Collect confidences per class (works even if boxes empty)
            confidences_by_class = {}
            for box in results.get("boxes", []):
                cls_map = results.get("class_map", CLASSES)
                cls_name = cls_map[int(box.cls)]
                # box.conf е тензор → изваждаме стойността правилно
                conf = float(box.conf.item() if hasattr(box.conf, "item") else box.conf)
                confidences_by_class.setdefault(cls_name, []).append(conf)

            # Build dataframe
            data_rows = []
            for cls_name, count in results.get("counts", {}).items():
                if isinstance(cls_name, int):
                    cls_name = CLASSES[cls_name]
                avg_conf = np.mean(confidences_by_class.get(cls_name, [])) if cls_name in confidences_by_class else 0
                data_rows.append({
                    "Class": cls_name,
                    "Count": count,
                    "Avg Confidence": f"{avg_conf:.2f}"
                })

            count_df = pd.DataFrame(data_rows)

            # <-- FIX: only sort/show if 'Count' exists and df not empty -->
            if not count_df.empty and "Count" in count_df.columns:
                st.dataframe(
                    count_df.sort_values("Count", ascending=False),
                    use_container_width=True,
                    height=min(400, 50 + 35 * len(count_df))
                )
            else:
                st.info("No objects detected in this image.")
                # optional: show an empty table with the expected columns
                st.dataframe(pd.DataFrame(columns=["Class", "Count", "Avg Confidence"]),
                            use_container_width=True,
                            height=100)

            # Show detection details in an expander (only if there are boxes)
            with st.expander("Detailed Detection Data"):
                if not results.get("boxes"):
                    st.write("No detection boxes available.")
                else:
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
        source_filter = st.selectbox(
            "Filter by source (optional)",
            [
                "",
                "Upload an image",
                "Paste from clipboard (text input)",
                "Paste from clipboard (button)"
            ]
        )
        
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
                    if image_data:
                        try:
                            decoded_image = base64.b64decode(image_data)
                            image = Image.open(io.BytesIO(decoded_image))
                            st.image(image, caption="Detected Image", use_container_width=True)

                            # --- Download button ---
                            st.download_button(
                                label="Download",
                                data=decoded_image,
                                file_name=doc.get("image_name", f"detection_{i+1}.jpg"),
                                mime="image/jpeg",
                                key=f"download_btn_{i}",  # unique key
                                use_container_width=False
                            )
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


    def get_all_collections():
        return db.list_collection_names()

    def get_detection_history(user_id, class_filter=None, source_filter=None, limit=100):
        results = []
        for col_name in get_all_collections():
            if col_name == "users":
                continue
            query = {"user_id": user_id}
            if class_filter:
                query["object_counts." + class_filter] = {"$exists": True}
            if source_filter:
                query["source"] = source_filter
            collection = db[col_name]
            docs = collection.find(query).sort("timestamp", -1).limit(limit)
            for doc in docs:
                doc["collection"] = col_name
                results.append(doc)
        return results

    def history_to_dataframe(docs):
        rows = []
        for doc in docs:
            timestamp = doc.get("timestamp")
            model_type = doc.get("model_type")
            source = doc.get("source")
            image_name = doc.get("image_name")
            collection = doc.get("collection", "")
            for cls, count in doc.get("object_counts", {}).items():
                rows.append({
                    "Date": timestamp,
                    "Class": cls,
                    "Count": count,
                    "Model": model_type,
                    "Source": source,
                    "Image": image_name,
                    "Collection": collection
                })
        return pd.DataFrame(rows)

    with tab4:
        st.subheader("Overall Statistics (All Detections)")


        user_id = st.session_state.user_id
        all_docs = get_detection_history(user_id=user_id, limit=500)  

        if not all_docs:
            st.warning("No detection history found for this user.")
        else:
            
            total_objects = 0
            class_counts = {}
            confidences_by_class = {}

            for doc in all_docs:
                counts = doc.get("object_counts", {})
                for cls, cnt in counts.items():
                    total_objects += cnt
                    class_counts[cls] = class_counts.get(cls, 0) + cnt

                for obj in doc.get("objects", []):
                    cls = obj["class"]
                    conf = obj["confidence"]
                    if cls not in confidences_by_class:
                        confidences_by_class[cls] = []
                    confidences_by_class[cls].append(conf)

            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown('<div class="metric-card"><div class="metric-title">Total Objects Detected</div>'
                            f'<div class="metric-value">{total_objects}</div></div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="metric-card"><div class="metric-title">Unique Classes</div>'
                            f'<div class="metric-value">{len(class_counts)}</div></div>', unsafe_allow_html=True)
            with col3:
                avg_conf = np.mean([c for lst in confidences_by_class.values() for c in lst]) if confidences_by_class else 0
                st.markdown('<div class="metric-card"><div class="metric-title">Average Confidence</div>'
                            f'<div class="metric-value">{avg_conf*100:.1f}%</div></div>', unsafe_allow_html=True)

            
            rows = []
            for cls, count in class_counts.items():
                avg_conf = np.mean(confidences_by_class.get(cls, [])) if cls in confidences_by_class else 0
                rows.append({
                    "Class": cls,
                    "Count": count,
                    "Avg Confidence": f"{avg_conf:.2f}"
                })

            df_overall = pd.DataFrame(rows)
            st.subheader("Object Counts Across All Detections")
            st.dataframe(
                df_overall.sort_values("Count", ascending=False),
                use_container_width=True,
                height=min(500, 50 + 35 * len(df_overall))
            )

            
            st.subheader("Class Distribution")
            st.bar_chart(df_overall.set_index("Class")["Count"])