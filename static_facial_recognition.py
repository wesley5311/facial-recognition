'''
Created on Dec 21, 2023

@author: Wesley
'''
import cv2 #recognition
from matplotlib import pyplot as plt #image display
import os #saving data
from pathlib import Path #necessary?
from tkinter import filedialog
#Thank you datacamp





file_paths = filedialog.askopenfilenames(
    initialdir="/",
    title="Select file to analyze",
    filetypes=[("Image files", [".jpg", ".png"])], #("Video files", [".mp4"])], #Removes need to check for valid filetype, program only allows jpg png
    multiple=False
    )

imagePath = file_paths[0]
img = cv2.imread(imagePath)
img_greyed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#create own classifier for different facial attributes#

#temporary cascade
face_class = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_class = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
#greyscale
face = face_class.detectMultiScale(
    img_greyed, scaleFactor = 1.1, minNeighbors=5, minSize=(40, 40)
)

##EYE CLASS HERE
eye = eye_class.detectMultiScale(
    img_greyed, scaleFactor = 1.1, minNeighbors=7, minSize = (10,10)
)
#return to rgb with box
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#box around face
for (x,y,w,h) in face:
    cv2.rectangle(img_rgb, (x,y), (x + w, y + h), (255,0,0), 4)

#box around eyes
for (x,y,w,h) in eye:
    cv2.rectangle(img_rgb, (x,y), (x + w, y + h), (0,255,0), 2)

#create window
plt.figure(figsize=(20,10))
plt.imshow(img_rgb)
plt.axis('off')
plt.show()

file_name = f'copy_of_{Path(imagePath).name}' #saves to workspace directory
cv2.imwrite(file_name, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)) 
print(f"Image saved as: {file_name}")
