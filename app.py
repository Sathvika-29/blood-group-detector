import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
import requests
import gdown


# --- Model Architecture (ResNet9) ---
class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = self.accuracy(out, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def accuracy(self, outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.network = torchvision.models.resnet18(pretrained=False)
        self.network.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.network.fc = nn.Linear(self.network.fc.in_features, num_classes)

    def forward(self, xb):
        return self.network(xb)

# --- Download model if not present ---
def download_model():
    model_path = "FingurePrintTOBloodGroup.pth"
    if not os.path.exists(model_path):
        # Replace this ID with your actual file ID
        file_id = "1X24hSd0qrdLkvspx-VtNgk43RwnA8mnB"
        url = f"https://drive.google.com/uc?id={1X24hSd0qrdLkvspx-VtNgk43RwnA8mnB}"
        gdown.download(url, model_path, quiet=False)
    

download_model()
# --- Load Model ---
model_path = 'FingurePrintTOBloodGroup.pth'
num_classes = 8
class_names = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ResNet9(3, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

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
