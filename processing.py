import numpy as np
import os
import cv2
import argparse
from tqdm import tqdm
import random



print(f"Preprocessing {1200} ...")

paths = os.listdir('../main/asl_alphabet_train/asl_alphabet_train')
paths.sort()
root = '../main/asl_alphabet_train/asl_alphabet_train'



for j, dir_path  in tqdm(enumerate(paths), total=len(paths)):
    all_pics = os.listdir(f"{root}/{dir_path}")
    os.makedirs(f"../main/preprocessed_image/{dir_path}", exist_ok=True)
    for i in range(1200): 
        id = (random.randint(0, 2999))
        pic = cv2.imread(f"{root}/{dir_path}/{all_pics[id]}")
        pic = cv2.resize(pic, (224, 224))
        cv2.imwrite(f"../main/preprocessed_image/{dir_path}/{dir_path}{i}.jpg", pic)
print('preprocessing complete.')
