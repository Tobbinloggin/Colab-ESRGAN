import os
source1 = "/content/Colab-ESRGAN/results"
dest11 = "/content/Colab-ESRGAN/results1"
dest21 = "/content/Colab-ESRGAN/results2"
files = os.listdir(source1)
import shutil
import numpy as np
amount_files = 17

files_counter = 0
for f in files:
    if 0.33*amount_files > files_counter:
      shutil.move(source1 + '/'+ f, dest11 + '/'+ f)
    if 0.33*amount_files < files_counter < 0.66*amount_files:
      shutil.move(source1 + '/'+ f, dest21 + '/'+ f)
    files_counter += 1