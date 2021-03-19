import torchvision.transforms as transforms
import cv2
import numpy
import numpy as np

from coco_names import COCO_INSTANCE_CATEGORY_NAMES as coco_names

# this will help us create a different color for each class
COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))

transform = transforms.Compose([
    transforms.ToTensor()
])

def predict(image, model, device, detection_threshold):
    # transform the image to tensor
    image = transform(image).to(device)
    image = image.unsqueeze(0)
    outputs = model(image)

    # print the results individually
    # print(f"BOXES: {outputs[0]['boxes']}")
    # print(f"LABELS: {outputs[0]['labels']}")
    # print(f"SCORES: {outputs[0]['scores']}")

    pred_classes = [coco_names[i] for i in outputs[0]['lables'].cpu().numpy()]

    #get score for all predicted objects
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()

    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()

