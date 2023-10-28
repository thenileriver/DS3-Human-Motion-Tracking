import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import pandas as pd
import os
import cv2 as cv
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.models as models
import torchvision.utils
import time
import turtle
import numpy as np

cam = cv.VideoCapture(0)

cv.namedWindow("test")

img_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv.imshow("test", frame)

    k = cv.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        print(frame)
        break

cam.release()

cv.destroyAllWindows()


#Hyper Paramaters
num_classes = 32
num_epochs = 20
batch_size = 20
learning_rate = 0.01
error = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Network(nn.Module):
    
    def __init__(self,num_classes=32):
        super().__init__()
        self.model_name='resnet101'
        self.model=models.resnet101()
        self.model.conv1=nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc=nn.Linear(self.model.fc.in_features, num_classes)
        
    def forward(self, x):
        x=self.model(x)
        return x

model = Network().to(device)

#Loss and Optimizer functions
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

loss_min = np.inf


#The path of the model
weights_path = 'HumanPose_resnet101.pth'

#The path of the image being plotted on
image_path = '030424224.jpg'

#These are the orignal landmarks for that image
#


best_network = Network()
best_network.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')), strict=False) 
best_network.eval()

image = frame
display_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
# image = display_image[y:h, w:x]
image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)


height, width = image.shape
Xratio = width/224
Yratio = height/224
image = TF.resize(Image.fromarray(image), size=(224, 224))
image = TF.to_tensor(image)
image = TF.normalize(image, [0.5], [0.5])

# plt.figure()
# plt.imshow(image.squeeze(0))
# plt.show()

#This gets the landmarks out of the model
with torch.no_grad():
    landmarks = best_network(image.unsqueeze(0))

image_x = landmarks[0][12].item() + (landmarks[0][12].item() - landmarks[0][14].item())/2   
image_y = landmarks[0][13].item() + (landmarks[0][13].item() - landmarks[0][15].item())/2 

correct = [[-(landmarks[0][i].item()-image_x)*Xratio,-(landmarks[0][i+1].item()-image_y)*Yratio] for i in range(0,len(landmarks[0]),2)]

screen = turtle.Screen()
screen.setup(700, 700)
screen.tracer(0)

skk = turtle.Turtle()
skk.width(3)
skk.speed('fastest')
skk.hideturtle()

def draw_stick_figure(coordinates):
    circle_radius = np.sqrt((coordinates[9][0] - coordinates[8][0]) ** 2 + (coordinates[9][1] - coordinates[8][1]) ** 2)/2
    circle_center = [(coordinates[9][0] + coordinates[8][0])/2,(coordinates[9][1] + coordinates[8][1])/2]
    angle = np.tanh((coordinates[9][1] - coordinates[8][1])/(coordinates[9][0] - coordinates[8][0]))
    
    skk.penup()
    skk.goto(coordinates[8])
    skk.pendown()
    skk.setheading(angle)
    skk.circle(circle_radius)
    
    skk.penup()
    skk.goto(coordinates[0])
    skk.dot()
    skk.pendown()
    skk.goto(coordinates[1])
    skk.dot()
    skk.goto(coordinates[2])
    skk.dot()
    skk.goto(coordinates[6])
    skk.dot()
    skk.goto(coordinates[3])
    skk.dot()
    skk.goto(coordinates[4])
    skk.dot()
    skk.goto(coordinates[5])
    skk.dot()
    skk.penup()
    skk.goto(coordinates[6])
    skk.pendown()
    skk.dot()
    skk.goto(coordinates[7])
    skk.dot()
    skk.goto(coordinates[8])
    skk.dot()
    skk.up()
    skk.goto(coordinates[9])
    skk.down()
    skk.dot()
    skk.penup()
    skk.goto(coordinates[10])
    skk.dot()
    skk.pendown()
    skk.goto(coordinates[11])
    skk.dot()
    skk.goto(coordinates[12])
    skk.dot()
    skk.goto(coordinates[7])
    skk.dot()
    skk.goto(coordinates[13])
    skk.dot()
    skk.goto(coordinates[14])
    skk.dot()
    skk.goto(coordinates[15])
    skk.dot()
    
draw_stick_figure(correct)
turtle.done()