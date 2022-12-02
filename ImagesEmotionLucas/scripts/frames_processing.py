#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import re
from emotion_recognition import FacialEmotionRecognition

def PlotEmotions(currentframe=None, max_scores=None):
    # Crear gráfica de variación de emociones a lo largo de un vídeo
    
    if currentframe and max_scores:
        # En x los frames
        x = np.array(range(currentframe))
        # En y las emociones 
        y = np.array(max_scores)

        plt.figure(figsize=(8,4))
        plt.ylim(-0.5,6.5)
        plt.plot(x,y, 's--r')
        plt.xlabel("Frames")
        plt.ylabel("Emotion")
        # plt.show()
        # Donde:
        # {0: 'Anger', 1: 'Disgust', 2: 'Fear', 3: 'Happiness', 4: 'Neutral', 5: 'Sadness', 6: 'Surprise'} 
    else:
        print("There is no data!")
    
def PlotEmotionsVariation(scores=None):
    
    if scores:
        emotion_scores = [[],[],[],[],[],[],[]]

        for i in range(7):
            for j in range(len(scores)):  
                emotion_scores[i].append(scores[j][i]) 

        # DataFrame
        data = {0: emotion_scores[0], 1: emotion_scores[1], 2: emotion_scores[2], 3: emotion_scores[3],
               4: emotion_scores[4], 5: emotion_scores[5], 6: emotion_scores[6]}
        df = pd.DataFrame(data)
        df.head()

        # Variación intra-emoción
        data_ar = np.array(df)

        plt.figure(figsize=(10,5))
        plt.plot(data_ar[:,0],'s--r',linewidth=1,color='b')
        plt.plot(data_ar[:,1],'s--r',linewidth=1,color='r')
        plt.plot(data_ar[:,2],'s--r',linewidth=1,color='g')
        plt.plot(data_ar[:,3],'s--r',linewidth=1,color='c')
        plt.plot(data_ar[:,4],'s--r',linewidth=1,color='m')
        plt.plot(data_ar[:,5],'s--r',linewidth=1,color='y')
        plt.plot(data_ar[:,6],'s--r',linewidth=1,color='orange')
        plt.gca().legend(('Anger','Disgust','Fear','Happiness','Neutral','Sadness','Surprise'))
        plt.xlabel("Frames")
        plt.ylabel("Scores")
        plt.show()
    else:
        print("There is no data!")

img_extensions=['.jpg','.jpeg','.png']
video_extensions=['.mov','.avi','.mp4']

def is_specialfile(path,exts):
    _, file_extension = os.path.splitext(path)
    return file_extension.lower() in exts

def is_image(path):
    return is_specialfile(path,img_extensions)

def is_video(path):
    return is_specialfile(path,video_extensions)

class ProcessFrame:
    
    def __init__(self, fpath=None, heat_map=False, want_plot=False, webcam=False):
        
        self.fpath = fpath
        self.heat_map = heat_map
        self.want_plot = want_plot
        self.webcam = webcam
        
        print(self.fpath)
        
        # Se decide si es una imagen, un conjunto de imágenes, o un vídeo
        # Comprobar si se trata de una imagen, un vídeo, o un conjunto de imágenes
        if webcam:
            ProcessFrame.Video(self)
        
        elif is_video(self.fpath):
            ProcessFrame.Video(self)
        
        elif is_image(self.fpath):
            ProcessFrame.Image(self)
        
        # Nos interesa una carpeta con imágenes
        else: ProcessFrame.FolderImages(self)
                
    def Image(self):
        # Extraemos la imagen
        frame=cv2.imread(self.fpath)

        # Llamamos a la clase de reconocimiento de emoción facial
        FER = FacialEmotionRecognition()
        ss_idk, sc_idk = FER.FramePrediction(frame=frame, heat_map=self.heat_map)
        
    def FolderImages(self):
        # Para procesar una carpeta entera con imágenes
        files_names = os.listdir(self.fpath)
        FER = FacialEmotionRecognition()
        cont_files = 0
        max_scores = []
        predictions = []
        scores = []

        # Ordenar los archivos de forma natural para que el 2 vaya después del 1, y no por ejemplo el 11
        r = re.compile(r"(\d+)")  
        files_names.sort(key=lambda x: int(r.search(x).group(1))) 

        # Creo una figure para representar
        fig = plt.figure(figsize=(10,6))  

        for file_name in files_names:

            image_path = self.fpath + "/" + file_name
            print(cont_files, image_path)

            # Extraemos la imagen
            frame = cv2.imread(image_path)

            # Añado subplots a la figure
            ax = fig.add_subplot(int(np.round(len(files_names)/4)),4,cont_files+1)  

            # Llamamos a la clase de reconocimiento de emoción facial 
            score, predicted_class = FER.FramePrediction(frame=frame, heat_map=self.heat_map)  

            # Título del subplot
            ax.set_title("Frame {}: {}".format(cont_files, predicted_class))   

            scores.append(score)
            max_scores.append(np.argmax(score))
            predictions.append(predicted_class)

            cont_files = cont_files + 1

        if self.want_plot:
            PlotEmotions(cont_files, max_scores)
            PlotEmotionsVariation(scores) # En Python este no se plotea
                
    def Video(self):
        
        if self.webcam:
            cam = cv2.VideoCapture(0)
        else: 
            cam = cv2.VideoCapture(self.fpath)

        fps = cam.get(cv2.CAP_PROP_FPS)
        print('frames per second =',fps)

        max_scores = []
        predictions = []
        scores = []

        currentframe = 0
        minutes = 0
        seconds = 0

        FER = FacialEmotionRecognition()

        while True:

            try:

                frame_id = int(fps*(minutes*60 + seconds))
                # print('frame id =',frame_id)

                cam.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ret, frame = cam.read()

                # Llamamos a la clase de reconocimiento de emoción facial
                score, predicted_class = FER.FramePrediction(frame=frame, heat_map=self.heat_map) 

                if score.any():
                    scores.append(score)
                    max_scores.append(np.argmax(score))
                    predictions.append(predicted_class)

                # Para contar los frames
                currentframe += 1

                # Cogemos un frame cada segundo
                seconds += 1

            except:
                cam.release()
                break
            
            # De momento no consigo que pare de grabar hasta que no detecta ninguna cara
            if cv2.waitKey(20) & 0xFF == ord('q'):        
                cam.release()
                cv2.destroyAllWindows()
                break
         
        if self.want_plot:
            PlotEmotions(currentframe, max_scores)
            PlotEmotionsVariation(scores) # En Python este no se plotea
                