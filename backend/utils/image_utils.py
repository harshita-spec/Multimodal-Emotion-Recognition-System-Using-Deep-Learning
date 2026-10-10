import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import io

# Haar cascades for face and smile detection
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
SMILE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_smile.xml'

face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
smile_cascade = cv2.CascadeClassifier(SMILE_CASCADE_PATH)

# Standard ImageNet normalization for ResNet/CNN architectures
image_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def detect_and_crop_face(image_bytes: bytes):
    """
    Detect face in image bytes, crop and align it, or return full resized image if no face detected.
    Returns:
        cropped_bgr_img (np.ndarray): BGR image matrix of cropped face (or full image resized).
        face_box (tuple): (x, y, w, h) bounding box.
        face_found (bool): Whether a face was detected.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    
    h, w, _ = img.shape
    if len(faces) > 0:
        # Pick largest face box
        faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
        (x, y, fw, fh) = faces[0]
        
        # Add slight padding around face ROI
        pad_w = int(fw * 0.15)
        pad_h = int(fh * 0.15)
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(w, x + fw + pad_w)
        y2 = min(h, y + fh + pad_h)
        
        face_crop = img[y1:y2, x1:x2]
        return face_crop, (x1, y1, x2 - x1, y2 - y1), True
    else:
        return img, (0, 0, w, h), False

def analyze_facial_heuristics(face_bgr: np.ndarray) -> np.ndarray:
    """
    Analyzes facial geometry (smile detection, mouth openness, brightness) to compute a 7-class emotion prior logit vector.
    Classes: ["Happy", "Sad", "Angry", "Fear", "Surprise", "Disgust", "Neutral"]
    """
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    
    # 1. Smile Detection in lower half of face
    lower_face_gray = gray[int(h * 0.5):h, :]
    smiles = smile_cascade.detectMultiScale(lower_face_gray, scaleFactor=1.7, minNeighbors=20, minSize=(25, 25))
    
    has_smile = len(smiles) > 0
    
    # 2. Mouth & Eyes Brightness / Openness Analysis
    mouth_roi = lower_face_gray[int(h * 0.15):, int(w * 0.2):int(w * 0.8)]
    mouth_std = np.std(mouth_roi) if mouth_roi.size > 0 else 0
    mouth_mean = np.mean(mouth_roi) if mouth_roi.size > 0 else 0
    
    # Overall facial luminance
    mean_val = np.mean(gray)
    std_val = np.std(gray)
    
    logits = np.zeros(7, dtype=np.float32)
    
    if has_smile or (mouth_std > 38 and mouth_mean > mean_val * 0.85):
        # Strongly indicate HAPPY
        logits[0] += 3.5  # Happy
        logits[4] += 0.8  # Surprise
        logits[6] += 0.2  # Neutral
    elif mouth_std > 48 and std_val > 45:
        # High variation (Open mouth / Wide eyes -> Surprise or Angry)
        logits[4] += 2.5  # Surprise
        logits[2] += 1.5  # Angry
        logits[3] += 1.0  # Fear
    elif mean_val < 75 and std_val < 25:
        # Dark / Low contrast facial region -> Sad
        logits[1] += 2.8  # Sad
        logits[6] += 1.2  # Neutral
        logits[5] += 0.5  # Disgust
    else:
        # Balanced facial expression -> Neutral or Slight Smile
        logits[6] += 2.0  # Neutral
        logits[0] += 1.0  # Happy
        logits[1] += 0.4  # Sad
        
    return logits

def preprocess_image_tensor(img_bgr: np.ndarray):
    """
    Converts BGR numpy image to normalized PyTorch tensor (1, 3, 224, 224).
    """
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    tensor = image_transform(pil_img).unsqueeze(0)
    return tensor
