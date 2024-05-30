from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50
import pickle
import cv2
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

feature_list = np.array(pickle.load(open('embedding.pkl','rb')))
filenames = pickle.load(open('filename.pkl','rb'))
detector = MTCNN()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = resnet50(pretrained=True)
model = nn.Sequential(*list(model.children())[:-2], nn.AdaptiveAvgPool2d((1, 1)))
model = model.to(device)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def img_to_vector(img, model):
    img = Image.fromarray(img)
    img_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        features = model(img_tensor)
    return features.cpu().numpy().flatten()

sample_img = cv2.imread('data\Aishwarya_Rai\Aishwarya_Rai.21.jpg')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
gray_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
if len(faces) == 1:
    x, y, width, height = tuple(faces[0])
    face = sample_img[y:y+height,x:x+width]
else:
    face = sample_img
result = img_to_vector(face, model)
ind = filenames.index('data\Aishwarya_Rai\Aishwarya_Rai.21.jpg')
print(result)
print(feature_list[ind])
# print(cosine_similarity(result.reshape(1, -1), feature_list[ind].reshape(1, -1)))


similarity = []
for i in range(len(feature_list)):
    si = cosine_similarity(result.reshape(1, -1), feature_list[i].reshape(1, -1))
    similarity.append(si[0][0])
index = sorted(list(enumerate(similarity)), reverse=True, key=lambda x:x[1])[0][0]
temp_img = cv2.imread(filenames[index])



