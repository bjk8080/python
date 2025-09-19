import numpy as np
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('TKAgg')

image= np.array([
    [0,0,0,0,0],   # ← 각 행 끝에 , 필요
    [0,0,1,0,0],
    [0,1,0,1,0],
    [0,0,1,0,0],
    [0,0,0,0,0]
])

plt.figure(figsize=(8,4))
plt.subplot(1,3,1)
plt.imshow(image,cmap='gray',vmin=0,vmax=1)
plt.title('original')