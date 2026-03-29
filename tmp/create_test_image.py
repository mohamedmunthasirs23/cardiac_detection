import requests
import numpy as np
import cv2
import json

# Create a dummy image
img = np.ones((500, 1000, 3), dtype=np.uint8) * 255 # White background
# Draw a sine wave
for x in range(1000):
    y = int(250 + 100 * np.sin(x / 20.0))
    cv2.circle(img, (x, y), 2, (0, 0, 0), -1)

cv2.imwrite('dummy_ecg.png', img)

print("Created dummy_ecg.png. Please test uploading this in the frontend if the server restarts.")
