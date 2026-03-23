# MULTIMODAL-FAKE-NEWS-DETECTION-USING-TEXT-ANALYSIS-AND-AI-GENERATED-IMAGE-IDENTIFICATION

A deep learning-based web application that detects whether a news article is **Real or Fake** using both **text and image inputs**. This project combines NLP and Computer Vision using **BERT + CNN + Fusion Model**.

---

## 🚀 Features

* 📝 **Text Analysis** using BERT embeddings
* 🖼️ **Image Analysis** using CNN (MobileNetV2)
* 🔗 **Multimodal Fusion Model** for final prediction
* ⚡ Fast and interactive **Streamlit Web UI**
* 📊 Confidence score for predictions

---

## 🧠 Model Architecture

* **Text Model:** BERT (CLS embeddings → Dense layers)
* **Image Model:** MobileNetV2 (feature extraction)
* **Fusion Model:** Concatenation of text + image features → Dense layers

---

## 📁 Project Structure

```
MultimodalFakeNewsDetection/
│
├── models/                  # Model files (not included in repo)
│   └── fusion_best_model.keras
│   └── text_only_model.keras
│   └── image_auth_model_torch.pth
│
├── UI/
│   └── main.py              # Streamlit app
│          
├── requirements.txt         # Dependencies
├── README.md
```

---

## ⚠️ Note on Models

Due to GitHub file size limits, trained models are not included.

👉 Download models from here:
📥 **[Google Drive Link](YOUR_LINK_HERE)**

After downloading, place them in:

```
models/
├── text_only_model.keras
├── fusion_model.keras
├── image_auth_model.pth
```

---

## ⚙️ Installation

### 1️⃣ Clone the repository

```
git clone https://github.com/your-username/MultimodalFakeNewsDetection.git
cd MultimodalFakeNewsDetection
```

### 2️⃣ Create virtual environment

```
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3️⃣ Install dependencies

```
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```
streamlit run UI/main.py
```

Then open:

```
http://localhost:8501
```

---

## 🧪 Usage

* Enter **news text**
* Upload an **image (optional)**
* Click **Analyze**
* Get prediction:

  * ✅ Real News
  * ❌ Fake News
* View confidence score

---

## 📊 Sample Output

```
Text: Real (53%)
Image: Fake (69%)
Final: Fake (97%)
```

---

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* PyTorch
* Transformers (BERT)
* Streamlit
* NumPy / PIL

---

## 🚀 Future Improvements

* 🔍 Explainable AI (attention visualization)
* 🌐 Full-stack deployment (FastAPI + React)
* ☁️ Cloud deployment (AWS / Render)
* 📱 Mobile-friendly UI

---

## 👨‍💻 Author

**P Sai Venkata Rajesh**
🎓 Final Year Student
💡 Interested in AI, ML, and Full Stack Development

---

## ⭐ If you like this project

Give it a ⭐ on GitHub and share your feedback!
