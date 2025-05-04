import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import torchvision.models as torchvision_models  # import torchvision.models as torchvision_models to avoid the conflict
import pydicom
import numpy as np

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Label map
label_map = {"Normal/Mild": 0, "Moderate": 1, "Severe": 2}
reverse_label_map = {v: k for k, v in label_map.items()}


# Define your custom model class
class CustomEfficientNetV2(nn.Module):
    def __init__(self, num_classes=3, pretrained_weights=True):
        super(CustomEfficientNetV2, self).__init__()
        self.model = models.efficientnet_v2_s(weights=True)
        if pretrained_weights:
            self.model.load_state_dict(torch.load(pretrained_weights))
        num_ftrs = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)


# Load all three trained models
def load_all_models(weights_path, device):
    model_paths = {
        "Sagittal T1": "sagittal_t1/best_model_epoch_1.pth",
        "Axial T2": "axial_t2/best_model_epoch_1.pth",
        "Sagittal T2/STIR": "saggittal_t2/best_model_epoch_1.pth",
    }

    loaded_models = {}
    for name, path in model_paths.items():
        model = CustomEfficientNetV2(num_classes=3, pretrained_weights=weights_path)
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device)
        model.eval()
        loaded_models[name] = model

    return loaded_models


# Image prediction function
def predict_image(model, image_path):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    # Use pydicom to load the DICOM image
    ds = pydicom.dcmread(image_path)
    image = ds.pixel_array
    # Convert the image to RGB
    image = np.stack((image,) * 3, axis=-1)
    image = Image.fromarray(image.astype(np.uint8)).convert(
        "RGB"
    )  # Convert to PIL Image
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = F.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probs, dim=1).item()

    return reverse_label_map[predicted_class], probs.squeeze().cpu().numpy()


# Plot probabilities
def plot_probabilities(model_name, probs):
    labels = list(label_map.keys())
    plt.figure(figsize=(6, 4))
    plt.bar(labels, probs, color="skyblue")
    plt.title(f"Prediction by {model_name}")
    plt.ylabel("Probability")
    plt.ylim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


# Main execution
if __name__ == "__main__":
    weights_path = "pre_trained_weights/efficientnet_v2_s-dd5fe13b.pth"
    image_path = "test_imgs/1.dcm"

    # Load the DICOM image using pydicom
    ds = pydicom.dcmread(image_path)
    image_data = ds.pixel_array

    plt.figure(figsize=(6, 6))
    plt.imshow(image_data, cmap="gray")
    plt.title("Input DICOM Image")
    plt.axis("off")
    plt.show()
    # Load all models
    all_models = load_all_models(weights_path, device)

    # For storing predictions
    results = []

    # Predict using all models
    for model_name, model in all_models.items():
        pred_class, prob_values = predict_image(model, image_path)
        print(f"\nModel: {model_name}")
        print(f"Predicted Class: {pred_class}")
        print(f"Probabilities: {prob_values}")
        plot_probabilities(model_name, prob_values)

        # Save to results list
        results.append(
            {
                "Model": model_name,
                "Predicted Class": pred_class,
                "Normal/Mild": prob_values[0],
                "Moderate": prob_values[1],
                "Severe": prob_values[2],
            }
        )

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv("model_predictions.csv", index=False)
    print("\n Predictions saved to 'model_predictions.csv'")
