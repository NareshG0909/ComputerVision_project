import streamlit as st
import cv2
import numpy as np
import joblib
from skimage.feature import hog
from skimage.color import rgb2gray
from PIL import Image
import tempfile

# Load the trained model and scaler
model = joblib.load("model/traditional_model.pkl")
scaler = joblib.load("model/scaler.pkl")
label_map = {0: 'cardboard', 1: 'glass', 2: 'metal', 3: 'paper', 4: 'plastic', 5: 'trash'}

# Feature extraction
def extract_features(img):
    img = cv2.resize(img, (128, 128))
    gray = rgb2gray(img)
    hog_features = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), visualize=False, block_norm='L2-Hys')
    hist_r = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
    hist_g = cv2.calcHist([img], [1], None, [256], [0, 256]).flatten()
    hist_b = cv2.calcHist([img], [2], None, [256], [0, 256]).flatten()
    hist_r /= hist_r.sum() if hist_r.sum() > 0 else 1
    hist_g /= hist_g.sum() if hist_g.sum() > 0 else 1
    hist_b /= hist_b.sum() if hist_b.sum() > 0 else 1
    return np.hstack([hog_features, hist_r, hist_g, hist_b])

# Detect objects
def detect_objects(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [cnt for cnt in contours if cv2.contourArea(cnt) > 500]

# Classify objects
def classify_image(uploaded_file):
    image = np.array(Image.open(uploaded_file).convert("RGB"))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    original = image.copy()
    contours = detect_objects(image)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        obj_region = image[y:y+h, x:x+w]
        if obj_region.size == 0: continue
        try:
            features = extract_features(obj_region)
            features_scaled = scaler.transform([features])
            prediction = model.predict(features_scaled)[0]
            label = label_map[prediction]
            cv2.rectangle(original, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(original, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (0, 255, 0), 2)
        except:
            continue

    return cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

# Streamlit UI
st.title("Garbage Classification - Multi Object Detection")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Original Image", use_column_width=True)
    result_img = classify_image(uploaded_file)
    st.image(result_img, caption="Classified Image", use_column_width=True)
