import os

base_path = '/home/alvaro/Downloads/AUTSL/val/val/'
for video in os.listdir(base_path):
    if 'depth.mp4' in video.split('_'):
        os.remove(base_path+video)
