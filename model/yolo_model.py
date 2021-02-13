import torch
import datetime

model = torch.hub.load('ultralytics/yolov3', 'yolov3', pretrained=True)
model # .cuda()
image = torch.zeros((1, 3, 640, 640)) # .cuda()

start = datetime.datetime.now()
output = model(image)
end = datetime.datetime.now()

print(end - start)