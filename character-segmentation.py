#THIS CODE EXECUTES THE CharacterSegmentation() FUNCTION FROM segment_words_into_characters.py for each word in out output folder of word segmentation script
import cv2
import numpy as np
import pandas as pd
import os
from segment_words_into_characters import CharacterSegmentation
OUTPUT_PATH = "path to folder containing output of word segmentation"

dirs = os.listdir(OUTPUT_PATH)
i=0
dirs.sort(key=int)
for d in dirs:
    path = os.path.join(OUTPUT_PATH,d)
    print(path)
    CharacterSegmentation(path)
