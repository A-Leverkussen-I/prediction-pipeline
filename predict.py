import argparse
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from PIL import Image

def build_model(num_classes, dropout_rate=0.5):
    """
    Reconstruct the model architecture identical to the training setup.
    """
    # Load a ResNet50 model (pretrained set to False since we'll load our own weights)
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    
    # Freeze layers except for 'layer4' and 'fc'
    for name, param in model.named_parameters():
        if "layer4" not in name and "fc" not in name:
            param.requires_grad = False

    # Get the input feature dimension for the fully connected layer
    num_ftrs = model.fc.in_features

    # Replace the fully connected layer with a custom classifier head
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout_rate),
        nn.Linear(512, num_classes)
    )
    
    return model

def load_image(image_path):
    """
    Load and preprocess an image from the given file path.
    """
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Error loading image: {e}")
    
    # Apply the transformation and add a batch dimension
    image = transform(image).unsqueeze(0)
    return image

def main():
    parser = argparse.ArgumentParser(description="Predict the class of an image using a trained ResNet50 model.")
    parser.add_argument("--model", required=True, help="Path to the trained model (.pth file).")
    parser.add_argument("--image", required=True, help="Path to the image file for prediction.")
    args = parser.parse_args()

    # Define the number of classes (must match your training setup)
    num_classes = 17
    
    # Define the class names list (ensure the order is identical to training)
    class_names = [
        "Celebration", "CrossedArms-45deg-l", "CrossedArms-45deg-r", "CrossedArms-90deg-l",
        "CrossedArms-90deg-r", "CrossedArms-frontal", "Full Body", "Half Body",
        "HandsOnHips-45deg-l", "HandsOnHips-45deg-r", "HandsOnHips-90-deg-l", "HandsOnHips-90deg-r",
        "Head Shot","Hero","HoldingBall", "HoldingBall-45deg-l", "HoldingBall-45deg-r"
    ]
    
    # Set device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Build the model and load the saved weights
    model = build_model(num_classes, dropout_rate=0.5)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model = model.to(device)
    model.eval()
    print("Model loaded successfully.\n")
    
    # Load and preprocess the image
    try:
        image = load_image(args.image)
    except Exception as e:
        print(e)
        return
    
    image = image.to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_index = predicted.item()
    
    predicted_label = class_names[predicted_index]
    print(f"Predicted class index: {predicted_index}")
    print(f"Predicted class label: {predicted_label}\n")

if __name__ == "__main__":
    main()