import json
import cv2
import numpy as np
import pandas as pd
from skimage import io,segmentation,morphology,measure,util

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

for name in registered["STATE"].unique():
    count_words = len(name.split(" "))
    entries = registered.loc[registered["STATE"].eq(name)]
    nation_territories = np.zeros_like(base,dtype=bool)
    name_color = np.array([255,255,255])
    nation_color = hex2color(config["color_land"])
    for i,series in entries.iterrows():
        row = int(series["PIN_ROW"])
        col = int(series["PIN_COL"])
        nation_territory = segmentation.flood(borders,(row,col),connectivity=1)
        nation_territories = np.logical_or(nation_territories,nation_territory)
        nation_land = np.logical_and(mask_lands,nation_territory)
        
        nation_color = hex2color(series["COLOR"])
        latest[nation_land] = nation_color
        if np.dot(nation_color,np.array([0.299,0.587,0.114]))>256/2:
            name_color = np.array([0,0,0])
    # Because we don't know if their territories are connected:
    nation_territories = morphology.binary_opening(nation_territories,footprint=np.ones((3,3)))
    for prop_territory in measure.regionprops(measure.label(nation_territories)):
        angle = prop_territory.orientation * 180 / np.pi - 90
        if angle < -90:
            angle = 180 + angle
        y,x = prop_territory.centroid
        x -= 0.35 * prop_territory.axis_major_length * np.cos(angle/180*np.pi)
        x = int(x)
        y += 0.35 * prop_territory.axis_major_length * np.sin(angle/180*np.pi)
        y = int(y)

        (name_width,name_height),_ = cv2.getTextSize(name,fontFace=cv2.FONT_HERSHEY_TRIPLEX,fontScale=1,thickness=1)
        scale = 0.8 * prop_territory.axis_major_length / name_width

        text = cv2.putText(
                           img=np.zeros_like(base,dtype=np.uint8), # Image.
                           text=name,	               # Text string to be drawn.
                           org=(x,y),	                   # Bottom-left corner of the text string in the image.
                           fontFace=cv2.FONT_HERSHEY_TRIPLEX, # Font type, see HersheyFonts.
                           fontScale=scale,	                   # Font scale factor that is multiplied by the font-specific base size.
                           color=255,	                       # Text color.
                           thickness=2,	                   # Thickness of the lines used to draw a text.
                           lineType=cv2.LINE_AA,	           # Line type. See LineTypes
                           bottomLeftOrigin=False,	           # When true, the image data origin is at the bottom-left corner. Otherwise, it is at the top-left corner.
                          )
        M = cv2.getRotationMatrix2D((x,y), angle, 1)
        text = cv2.warpAffine(text, M, (text.shape[1], text.shape[0]))
        latest[(text>255/2)] = name_color

latest[(borders>0)] = np.array([255,255,255])
io.imsave("latest.png",util.img_as_ubyte(latest))
