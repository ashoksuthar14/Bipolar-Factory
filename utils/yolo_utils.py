import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import cv2

# Your class names from data.yaml
class_labels = ["Bird", "Drone", "Helicopter", "Missile", "Plane"]

def load_model(weights_path, device='cpu', num_classes=5, anchors=3):
    from torchvision import models
    import torch.nn as nn

    class YOLOStyleEfficientNet(nn.Module):
        def __init__(self, num_classes=5, num_anchors=3):
            super(YOLOStyleEfficientNet, self).__init__()
            efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            self.backbone = efficientnet.features
            self.detector = nn.Sequential(
                nn.Conv2d(1280, 512, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(512, num_anchors * (5 + num_classes), kernel_size=1)
            )

        def forward(self, x):
            x = self.backbone(x)
            x = self.detector(x)
            return x

    model = YOLOStyleEfficientNet(num_classes=num_classes, num_anchors=anchors)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    tensor = transform(image)
    return image, tensor.unsqueeze(0)  # Return original PIL and batched tensor

def predict_and_draw(image_pil, model, device='cpu', num_classes=5, anchors=3, threshold=0.4):
    image_tensor = transforms.ToTensor()(image_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        preds = model(image_tensor)  # Shape: [1, C, H, W]

    batch, channels, grid_h, grid_w = preds.shape
    preds = preds[0].permute(1, 2, 0)  # → [H, W, C]
    preds = preds.reshape(grid_h, grid_w, anchors, 5 + num_classes).cpu().numpy()

    image_np = np.array(image_pil)
    h, w, _ = image_np.shape

    detections = []

    for i in range(grid_h):
        for j in range(grid_w):
            for a in range(anchors):
                pred = preds[i, j, a]
                obj_score = 1 / (1 + np.exp(-pred[4]))  # Sigmoid
                if obj_score > threshold:
                    cx, cy, bw, bh = pred[0:4]
                    class_id = np.argmax(pred[5:])
                    label = class_labels[class_id] if class_id < len(class_labels) else f"Class {class_id}"
                    detections.append(f"{label} ({obj_score:.2f})")

                    # Convert center coords to box corners
                    x_center = (j + cx) / grid_w * w
                    y_center = (i + cy) / grid_h * h
                    width = bw * w
                    height = bh * h

                    x1 = int(x_center - width / 2)
                    y1 = int(y_center - height / 2)
                    x2 = int(x_center + width / 2)
                    y2 = int(y_center + height / 2)

                    cv2.rectangle(image_np, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(image_np, f"{label} ({obj_score:.2f})", (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return Image.fromarray(image_np), detections
