import numpy as np
from skimage import io,measure,draw,util

rad = 50

dots = io.imread("images/border_dots.gif")
height,width = dots.shape
lines = np.zeros_like(dots,dtype=int)

labels = measure.label(dots)
properties = measure.regionprops(labels)
for prop in properties:
    row,col = [int(i) for i in prop.centroid]
    
    row_min = max(0,     row-rad)
    row_max = min(height,row+rad)
    col_min = max(0,     col-rad)
    col_max = min(width, col+rad)

    window = labels[row_min:row_max,col_min:col_max]
    unique = np.unique(window[np.logical_and(window!=prop.label,window!=0)])
    if len(unique) < 2:
        continue
    distance = np.zeros_like(unique,dtype=float)
    for u,uni in enumerate(unique):
        distance[u] = np.sum(np.square(
            np.array(prop.centroid)
          - np.array(properties[uni-1].centroid)
        ))

    arg_dist = np.argsort(distance)
    dest0 = [int(i) for i in properties[unique[arg_dist[0]]-1].centroid]
    dest1 = [int(i) for i in properties[unique[arg_dist[1]]-1].centroid]

    rr0,cc0 = draw.line(row,col,*dest0)
    rr1,cc1 = draw.line(row,col,*dest1)

    lines[rr0,cc0] = 255
    lines[rr1,cc1] = 255

io.imsave("images/border_lines.gif",util.img_as_ubyte(lines))
