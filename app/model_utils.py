# app/model_utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import pydicom
import numpy as np
from io import BytesIO

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

label_map = {"Normal/Mild": 0, "Moderate": 1, "Severe": 2}
reverse_label_map = {v: k for k, v in label_map.items()}


class CustomEfficientNetV2(nn.Module):
    def __init__(self, num_classes=3):
        super(CustomEfficientNetV2, self).__init__()
        self.model = models.efficientnet_v2_s(weights=None)
        num_ftrs = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)


def load_all_models():
    model_paths = {
        "Sagittal T1": "sagittal_t1/best_model_epoch_1.pth",
        "Axial T2": "axial_t2/best_model_epoch_1.pth",
        "Sagittal T2/STIR": "saggittal_t2/best_model_epoch_1.pth",
    }

    models_dict = {}
    for name, path in model_paths.items():
        model = CustomEfficientNetV2(num_classes=3)
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        models_dict[name] = model
    return models_dict


def preprocess_dicom(dicom_bytes):
    ds = pydicom.dcmread(BytesIO(dicom_bytes))
    image = ds.pixel_array
    image = np.stack((image,) * 3, axis=-1)
    image = Image.fromarray(image.astype(np.uint8)).convert("RGB")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return transform(image).unsqueeze(0).to(device)


def predict_all_models(image_tensor, models_dict):
    results = {}
    with torch.no_grad():
        for model_name, model in models_dict.items():
            outputs = model(image_tensor)
            probs = F.softmax(outputs, dim=1).squeeze().cpu().numpy()
            prob_dict = {label: float(probs[idx]) for label, idx in label_map.items()}
            results[model_name] = {"Probabilities": prob_dict}
    return results
