#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 17:16:13 2024

@author: mithu
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt


image_path = '/Users/mithu/Documents/GitHub/Project_3-/motherboard_image.JPEG'
image = cv2.imread(image_path)


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 1. Using Canny Edge Detection
edges = cv2.Canny(gray, threshold1=50, threshold2=150)


# 2. Dilate edges to close gaps and connect contours
kernel = np.ones((5, 5), np.uint8)
dilated_edges = cv2.dilate(edges, kernel, iterations=1)

# 3. Find contours from the dilated edges
contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 4. Find the largest contour 
largest_contour = max(contours, key=cv2.contourArea)

# 5. Create a mask for the motherboard
mask = np.zeros_like(gray)
cv2.drawContours(mask, [largest_contour], -1, (255), thickness=cv2.FILLED)

# 6. Apply the mask to the original image
masked_image = cv2.bitwise_and(image, image, mask=mask)

# 7. Display the masked image
masked_image_rgb = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(8, 6))
plt.imshow(masked_image_rgb)
plt.title("Motherboard Only")
plt.axis('off')
plt.show()