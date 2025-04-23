import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import os
import gdown

# --- Download model if not present ---
def download_model():
    model_path = "FingurePrintTOBloodGroup.pth"
    if not os.path.exists(model_path):
        file_id = "1X24hSd0qrdLkvspx-VtNgk43RwnA8mnB"  # Update with your real ID if changed
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, model_path, quiet=False)
    return model_path

model_path = download_model()

# --- Load Full Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(model_path, map_location=device)
model.to(device)
model.eval()

# --- Constants ---
class_names = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]

# --- Image Transformations ---
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# --- Streamlit App ---
st.title("🩸 Fingerprint to Blood Group Detector")
st.write("Upload a fingerprint image to predict the blood group.")

uploaded_file = st.file_uploader("Choose a fingerprint image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1).squeeze()
        predicted_index = torch.argmax(probabilities).item()
        predicted_class = class_names[predicted_index]
        confidence = probabilities[predicted_index].item() * 100

    st.success(f"**Predicted Blood Group:** {predicted_class}")
    st.info(f"Model Confidence: {confidence:.2f}%")
    st.info("Model Accuracy: ~92% (as per training results)")

st.markdown("---")
st.caption("Developed with ❤️ using PyTorch and Streamlit")

