import json
import numpy as np
import pandas as pd
from skimage import io,segmentation,util

def hex2color(hex:str):
    assert hex[0] == "#", "ColorError: lost `#` before hex code."
    hex = hex[1:]
    return np.array([int(hex[h:h+2],16) for h in (0,2,4)])

registered = pd.read_csv("REGISTER.csv")
registered = registered.dropna(subset="COLOR")

config = json.load(open("config.json",'r'))

base = io.imread("images/base.gif")
mask_lands = (base > 0)
mask_ocean = np.logical_not(mask_lands)
borders = io.imread("images/border_lines.gif")

latest = np.zeros((*base.shape,3),dtype=int)
latest[mask_lands] = hex2color(config["color_land"])
latest[mask_ocean] = hex2color(config["color_seas"])

for i,series in registered.iterrows():
    row = int(series["PIN_ROW"])
    col = int(series["PIN_COL"])
    mask_territory = segmentation.flood(borders,(row,col),connectivity=1)
    latest[np.logical_and(mask_lands,mask_territory)] = hex2color(series["COLOR"])

latest[(borders>0)] = np.array([255,255,255])
io.imsave("latest.png",util.img_as_ubyte(latest))
