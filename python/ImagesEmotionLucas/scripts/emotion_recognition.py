#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import tensorflow as tf

from tensorflow.compat.v1.keras.backend import set_session
from PIL import Image
from PIL import Image
from torchvision import transforms

# Importar módulo facial_analysis.py
from .facial_analysis import FacialImageProcessing

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from IPython import get_ipython

# get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings("ignore")


class FacialEmotionRecognition:

    def __init__(self):

        # get_ipython().run_line_magic('env', 'CUDA_VISIBLE_DEVICES=0')

        # Configuración de la sesión
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.compat.v1.Session(config=config)
        set_session(sess)

        # Comprobar si hay cuda o si se ha de usar la cpu
        self.use_cuda = torch.cuda.is_available()
        print("Cuda: ", self.use_cuda)
        self.device = 'cuda' if self.use_cuda else 'cpu'

        # Utilización modelo de procesado de la imagen
        self.imgProcessing = FacialImageProcessing(False)

        self.IMG_SIZE = 260
        self.image_to_tensor_transform = transforms.Compose(
            [
                transforms.Resize((self.IMG_SIZE, self.IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ]
        )

        # El modelo escogido es enet_b2_7.pt, que predice las 7 emociones básicas
        NUM_EMOTIONS = 7

        models_path, _ = os.path.split(os.path.realpath(__file__))

        PATH = os.path.join(models_path, 'models',
                            'affectnet_emotions', 'enet_b2_7.pt')

        self.idx_to_class = {0: 'Anger', 1: 'Disgust', 2: 'Fear',
                             3: 'happiness', 4: 'Neutral', 5: 'Sadness', 6: 'Surprise'}

        self.class_array = self.idx_to_class.values()

        # Transformar la imagen a tensor para hacer inferencia
        # print(PATH)
        self.model = torch.load(PATH, map_location=torch.device(
            self.device))  # Cargar el modelo en la cpu
        self.model = self.model.to(self.device)
        # model.eval()

    def predict_frame(self, frame=None, heat_map=False):

        scores = [-1000, -1000,  -1000, -1000,  -1000,  -1000,  -1000]
        predicted_class = None

        # Comprobar si hay frame
        if frame is not None:

            # Predicción de las caras de la imagen y plot de las mismas junto a sus predicciones
            frame_bgr = frame
            # plt.figure(figsize=(5, 5))
            frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            # plt.axis('off')
            # plt.imshow(frame)
            bounding_boxes, points = self.imgProcessing.detect_faces(frame)
            points = points.T

            cont_faces = 1

            # Comprobar si hay caras
            if bounding_boxes.any():

                for bbox, p in zip(bounding_boxes, points):
                    try:
                        box = bbox.astype(np.int)
                        x1, y1, x2, y2 = box[0:4]

                        # x1 = max(x1, 0)
                        # y1 = max(y1, 0)
                        # x2 = min(x2, self.IMG_SIZE)
                        # y2 = min(y2, self.IMG_SIZE)

                        face_img = frame[y1:y2, x1:x2, :]

                        img_tensor = self.image_to_tensor_transform(
                            Image.fromarray(face_img))
                        img_tensor.unsqueeze_(0)
                        scores = self.model(img_tensor.to(self.device))
                        scores = scores[0].data.cpu().numpy()
                        predicted_class = self.idx_to_class[np.argmax(scores)]

                        # plt.figure(figsize=(3, 3))
                        # plt.axis('off')
                        # plt.imshow(frame)
                        # plt.title(predicted_class)
                        # print("Scores cara {}: ".format(cont_faces), scores)
                        # print("Predicted class: ", predicted_class)

                        cont_faces = cont_faces+1

                        # Para ver el mapa de calor
                        if heat_map == True:

                            # Para ver el mapa de calor (GradCAM), que ayuda a detectar las regiones
                            # consideradas importantes por la red neuronal para realizar la predicción

                            target_layers = [self.model.blocks[-1][-1]]
                            # Construct the CAM object once, and then re-use it on many images:
                            cam = GradCAM(
                                model=self.model, target_layers=target_layers, use_cuda=self.use_cuda)

                            grayscale_cam = cam(input_tensor=img_tensor)
                            grayscale_cam = grayscale_cam[0, :]
                            face_img = cv2.resize(
                                face_img, (self.IMG_SIZE, self.IMG_SIZE))
                            rgb_img = np.float32(face_img) / 255
                            visualization = show_cam_on_image(
                                rgb_img, grayscale_cam, use_rgb=True)

                            plt.figure(figsize=(3, 3))
                            plt.axis('off')
                            plt.imshow(visualization)
                            plt.title(predicted_class)
                            # plt.show()

                    except:
                        print("There was a bounding frame error - skipping frame")
                        # Si no hay caras
            else:
                print("There are no faces!")

        # Si no hay imagen (frame)
        else:
            print("There is no image!")

        return dict((zip(self.class_array, scores))), predicted_class
