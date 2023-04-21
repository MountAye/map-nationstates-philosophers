import numpy as np
import pandas as pd
from skimage import io,transform,util

new_north2old_north = pd.DataFrame({
    "new_north_x": [2115,2425,2180,2905,2800,2975],
    "new_north_y": [ 187, 390, 655, 625, 965,1415],
    "old_north_x": [ 185, 450, 145, 875, 700, 775],
    "old_north_y": [ 187, 390, 655, 625, 965,1415],
})
new_north2new_south = pd.DataFrame({
    "new_north_x": []
    "new_north_y": []
    "old_north_x": []
    "old_north_y": []
})