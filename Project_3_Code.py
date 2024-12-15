#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 17:16:13 2024

@author: mithu
"""
#part 2
import cv2
import numpy as np
from matplotlib import pyplot as plt


image_path = '/Users/mithu/Documents/GitHub/Project_3-/motherboard_image.JPEG'
image = cv2.imread(image_path)


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Using Canny Edge Detection
edges = cv2.Canny(gray, threshold1=50, threshold2=150)


# Dilate edges to close gaps and connect contours
kernel = np.ones((5, 5), np.uint8)
dilated_edges = cv2.dilate(edges, kernel, iterations=1)

#  Find contours from the dilated edges
contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#  Find the largest contour 
largest_contour = max(contours, key=cv2.contourArea)

# Create a mask for the motherboard
mask = np.zeros_like(gray)
cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)

# Apply the mask to the original image
masked_image = cv2.bitwise_and(image, image, mask=mask)


# Display the masked image
masked_image_rgb = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(8, 6))
plt.imshow(masked_image_rgb)
plt.title("Motherboard Only")
plt.axis('off')
plt.show()

#Part 2
#!pip install ultralytics 
from google.colab import drive
from ultralytics import YOLO
import torch
drive.mount('/content/drive')

# dataset path
dataset_path = '/content/drive/MyDrive/data/data.yaml'
evaluation = '/content/drive/MyDrive/data/evaluation'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = YOLO('yolov8n.pt') #Load the model

#train the model
results = model.train(data=dataset_path,
                      epochs=100,
                      batch = 4,
                      imgsz = 1000,
                      device = device)
#Validation
results = model.val(data = dataset_path)
#Evaluation of the model
results = model.predict(source=evaluation, save=True)