#!/usr/bin/env python
# coding: utf-8
from frames_processing import ProcessFrame

# Pruebas con:

# 1) Una imagen
# fpath = r'C:\Users\lucas\OneDrive - UPV\Diploma AI Samsung\Project\ImagesEmotionLucas\test_images\lucas\WIN_20221129_14_17_56_Pro.jpg'
# ProcessFrame(fpath)

# 2) Carpeta con imágenes
fpath = r'C:\Users\lucas\OneDrive - UPV\Diploma AI Samsung\Project\ImagesEmotionLucas\test_images\sample_input'
ProcessFrame(fpath, want_plot = True)

# 3) Vídeo
# fpath = r'C:\Users\lucas\OneDrive - UPV\Diploma AI Samsung\Project\ImagesEmotionLucas\test_images\WIN_20221201_19_23_14_Pro.mp4'
# ProcessFrame(fpath, webcam=True, want_plot=True)










