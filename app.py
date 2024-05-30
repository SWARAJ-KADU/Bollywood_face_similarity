import streamlit as st
from PIL import Image
import os
import cv2, torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

feature_list = np.array(pickle.load(open('embedding.pkl','rb')))
filenames = pickle.load(open('filename.pkl','rb'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

model = resnet50(pretrained=True)
# Remove the classification layer and add global average pooling
model = nn.Sequential(*list(model.children())[:-2], nn.AdaptiveAvgPool2d((1, 1)))
model = model.to(device)

# Define the image transformations
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def save_uploaded_image(img):
    try:
        with open(os.path.join('uploads', img.name), 'wb') as f:
            f.write(img.getbuffer())
        return True
    except:
        return False
        
def img_to_vector(img, model):
    img = Image.fromarray(img)
    img_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension
    img_tensor = img_tensor.to(device)
    with torch.no_grad():
        features = model(img_tensor)
    return features.cpu().numpy().flatten()

def extract_feature(img_path, model):
    sample_img = cv2.imread(img_path)
    gray_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 1:
        x, y, width, height = tuple(faces[0])
        face = sample_img[y:y+height,x:x+width]
    else:
        face = sample_img
    result = img_to_vector(face, model)
    return result

def recommend(feature):
    similarity = []
    for i in range(len(feature_list)):
        si = cosine_similarity(feature.reshape(1, -1), feature_list[i].reshape(1, -1))
        similarity.append(si[0][0])
    index = sorted(list(enumerate(similarity)), reverse=True, key=lambda x:x[1])[0][0]
    return index

st.title('Which bollywood celebrity are you?')

uploaded_img = st.file_uploader('Choose an image')
if uploaded_img is not None:
    if save_uploaded_image(uploaded_img):
        display_image = Image.open(uploaded_img)
        st.image(display_image)
        features = extract_feature(os.path.join('uploads', uploaded_img.name), model)
        index = recommend(features)
        col1, col2 = st.beta_columns(2)
        with col1:
            st.header('Your uploaded immage')
            st.image(display_image)
        with col2:
            pred = " ".join(filenames[index].split('\\')[1].split('_'))
            st.header('Seems like'+pred)
            st.image(filenames[index], width = 300)


