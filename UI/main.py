import streamlit as st
from PIL import Image
import tempfile
import torch
import torch.nn as nn
from tensorflow.keras.models import load_model
from transformers import BertTokenizer, BertModel
from torchvision import transforms
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
import numpy as np
import tensorflow as tf

mobilenet = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

st.title("📰 Multimodal Fake News Detector")

def load_all_models():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ---- TensorFlow models ----
    text_model = load_model("../models/text_only_model.keras")
    fusion_model = load_model("../models/fusion_best_model.keras")
    IMAGE_MODEL_PATH  = "../models/image_auth_model_torch.pth"
    # ---- PyTorch image model ----
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 28 * 28, 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, 1)
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    image_model = SimpleCNN().to(DEVICE)
    image_model.load_state_dict(torch.load(IMAGE_MODEL_PATH, map_location=DEVICE))
    image_model.eval()

    # ---- BERT ----
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert = BertModel.from_pretrained("bert-base-uncased").to(DEVICE)
    bert.eval()

    mobilenet = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg"
    )


    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    return text_model, image_model, fusion_model, tokenizer, bert, DEVICE, mobilenet, transform

@st.cache_resource
def load_models():
    return load_all_models()

text_model, image_model, fusion_model, tokenizer, bert, DEVICE, mobilenet, transform = load_models()


def predict_text_only(text):
    with torch.no_grad():
        tokens = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        ).to(DEVICE)

        output = bert(**tokens)
        text_emb = output.last_hidden_state[:, 0, :].cpu().numpy()

    prob = text_model.predict(text_emb, verbose=0)[0][0]

    return {
        "prediction": "Fake News" if prob >= 0.5 else "Real News",
        "confidence": f"{prob*100:.2f}%" if prob >= 0.5 else f"{(1-prob)*100:.2f}%"
    }

def predict_image_only(image_path):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prob = torch.sigmoid(image_model(img)).item()

    return {
        "image_prediction": "AI-generated" if prob >= 0.5 else "Authentic",
        "confidence": f"{prob*100:.2f}%" if prob >= 0.5 else f"{(1-prob)*100:.2f}%"
    }

def predict_text_and_image(text, image_path):
    # ---- Text embedding ----
    with torch.no_grad():
        tokens = tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        ).to(DEVICE)

        output = bert(**tokens)
        text_emb = output.last_hidden_state[:, 0, :].cpu().numpy()

    # ---- Image embedding (ResNet-style) ----
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    mobilenet = tf.keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg"
    )

    img_emb = mobilenet.predict(x, verbose=0)

    # ---- Fusion ----
    fusion_input = np.concatenate([text_emb, img_emb], axis=1)
    prob = fusion_model.predict(fusion_input, verbose=0)[0][0]

    return {
        "news_prediction": "Fake News" if prob >= 0.5 else "Real News",
        "confidence": f"{prob*100:.2f}%" if prob >= 0.5 else f"{(1-prob)*100:.2f}%"
    }




# Input
text = st.text_area("Enter News Text")
image_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

# Predict button
if st.button("Analyze"):
    image_path = None

    if image_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(image_file.read())
            image_path = tmp.name

    # Call your function
    if text and image_path:
        result = predict_text_and_image(text, image_path)
    elif text:
        result = predict_text_only(text)
    elif image_path:
        result = predict_image_only(image_path)
    else:
        st.warning("Please provide text or image")
        st.stop()

    # Output
    st.subheader("Result")
    st.write(result)

    # Better UI
    if "confidence" in result:
        conf = float(result["confidence"].replace("%",""))
        st.progress(conf / 100)