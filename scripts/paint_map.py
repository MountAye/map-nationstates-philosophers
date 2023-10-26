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

mask_nations = np.zeros((*base.shape,3),dtype=int)

for _,territroy in registered.dropna(subset="COLOR").iterrows():
    row = int(territroy["PIN_ROW"])
    col = int(territroy["PIN_COL"])
    territory_all = segmentation.flood(borders,(row,col),connectivity=1)
    territroy_land = np.logical_and(mask_lands,territory_all)
    territroy_color = hex2color(territroy["COLOR"])
    latest[territroy_land] = territroy_color
    mask_nations[morphology.binary_dilation(territory_all,footprint=np.ones((3,3),dtype=int))] = territroy_color
print("PAINTED COLORS")

named = latest.copy()
named[(borders>0)] = np.array([255,255,255])
for n,name in enumerate(registered["STATE"].unique()):
    entries = registered.loc[registered["STATE"].eq(name)]
    nation_color = hex2color(entries.loc[entries.index[0],"COLOR"])
    nation_territories = np.all(mask_nations==nation_color,axis=-1)
    # Because we don't know if their territories are connected:
    nation_territories_label = measure.label(nation_territories)
    for prop_territory_all in measure.regionprops(nation_territories_label):
        # prop_territory_land = measure.regionprops(util.img_as_ubyte(np.logical_and(mask_lands,(nation_territories_label==prop_territory_all.label))))[0]
        angle = prop_territory_all.orientation * 180 / np.pi - 90
        if angle < -90:
            angle = 180 + angle


        words = name.split("\ ")
        n_words = len(words)
        longest = sorted(words,key=lambda x:len(x),reverse=True)[0]

        (name_width,name_height),_ = cv2.getTextSize(longest,fontFace=cv2.FONT_HERSHEY_TRIPLEX,fontScale=1,thickness=1)
        scale_x = 0.8 * prop_territory_all.axis_major_length / name_width
        scale_y = 0.8 * prop_territory_all.axis_minor_length / name_height / n_words
        scale = min([scale_x,scale_y])

        y,x = prop_territory_all.centroid
        x -= 0.45 * name_width * scale * np.cos(angle/180*np.pi)
        y += 0.45 * name_width * scale * np.sin(angle/180*np.pi)
        x = int(x)
        y = int(y)

        for w,word in enumerate(words):
            xw = int(x + scale * name_height * 1.2 * (w - (n_words-1)/2) * np.sin(angle/180*np.pi)) 
            yw = int(y + scale * name_height * 1.2 * (w - (n_words-1)/2) * np.cos(angle/180*np.pi)) 
            
            text_core = cv2.putText(
                               img=np.zeros_like(base,dtype=np.uint8), # Image.
                               text=word,	                           # Text string to be drawn.
                               org=(xw,yw),	                           # Bottom-left corner of the text string in the image.
                               fontFace=cv2.FONT_HERSHEY_TRIPLEX,      # Font type, see HersheyFonts.
                               fontScale=scale,	                       # Font scale factor that is multiplied by the font-specific base size.
                               color=255,	                           # Text color.
                               thickness=2,	                           # Thickness of the lines used to draw a text.
                               lineType=cv2.LINE_AA,	               # Line type. See LineTypes
                               bottomLeftOrigin=False,	               # When true, the image data origin is at the bottom-left corner. Otherwise, it is at the top-left corner.
                              )
            M = cv2.getRotationMatrix2D((xw,yw), angle, 1)
            text_core = cv2.warpAffine(text_core, M, (text_core.shape[1], text_core.shape[0]))
            text_core = (text_core>255/2)
            text_edge = morphology.binary_dilation(text_core,footprint=np.ones((5,5)))
            named[text_edge] = np.array([  0,  0,  0])
            named[text_core] = np.array([255,255,255])
    print(f"TAGGING NAMES #{n+1}: {name}")

io.imsave("latest.png",util.img_as_ubyte(named))

