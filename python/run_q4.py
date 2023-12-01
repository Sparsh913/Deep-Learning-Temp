import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.io
import skimage.filters
import skimage.morphology
import skimage.segmentation

from nn import *
from q4 import *
# do not include any more libraries here!
# no opencv, no sklearn, etc!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

for img in os.listdir("D:/CMU/F23/16-720/HW5/images"):
    im1 = skimage.img_as_float(skimage.io.imread(os.path.join('D:/CMU/F23/16-720/HW5/images',img)))
    bboxes, bw = findLetters(im1)
    print("Shape of bw: ", bw.shape)
    print("Shape of bboxes: ", len(bboxes))

    plt.imshow(bw)
    for bbox in bboxes:
        minr, minc, maxr, maxc = bbox
        rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                fill=False, edgecolor='red', linewidth=1)
        plt.gca().add_patch(rect)
    plt.show()
    # find the rows using..RANSAC, counting, clustering, etc.
    ##########################
    ##### your code here #####
    ##########################
    ht = [bbox[2] - bbox[0] for bbox in bboxes] # height of each bounding box
    ht = np.array(ht) 
    print("Shape of ht: ", ht.shape)
    avg_ht = np.mean(ht) # average height
    print("Average height: ", avg_ht) 
    mid_pts = [((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2, bbox[2] - bbox[0], bbox[3] - bbox[1])  for bbox in bboxes] # mid points of each bounding box
    mid_pts = sorted(mid_pts, key=lambda x: x[0]) # sort by x coordinate
    r = [] # rows
    r_total = [] # total rows
    h_prev = mid_pts[0][0] # previous height
    for mid_pt in mid_pts: # iterate through mid points
        if (mid_pt[0] > avg_ht + h_prev): 
            r = sorted(r, key=lambda x: x[1]) # sort by y coordinate
            r_total.append(r)
            r = [mid_pt] # reset row
            h_prev = mid_pt[0] # reset previous height
        else:
            r.append(mid_pt)
    r = sorted(r, key=lambda x: x[1]) # sort by y coordinate
    r_total.append(r)

    # crop the bounding boxes
    # note.. before you flatten, transpose the image (that's how the dataset is!)
    # consider doing a square crop, and even using np.pad() to get your images looking more like the dataset
    ##########################
    ##### your code here #####
    ##########################
    kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]) # kernel for dilation
    data = []
    for r in r_total:
        row_info = []
        for y, x, h, w in r:
            crop = bw[y-h//2:y+h//2, x-w//2:x+w//2] # crop the bounding box

            pad_h, pad_w = 0, 0
            if h > w:
                pad_h = h//20 # padding
                w_pad = (h - w)//2 + pad_h # padding
            elif w > h:
                pad_w = w//20
                h_pad = (w - h)//2 + pad_w
            crop = np.pad(crop, ((pad_h, pad_h), (pad_w, pad_w)), 'constant', constant_values=(1, 1)) # pad the image

            crop = skimage.transform.resize(crop, (32, 32)) # resize the image
            crop = skimage.morphology.erosion(crop, kernel) # erode the image
            crop = crop.T.flatten() # flatten the image
            row_info.append(crop) # append the crop
        data.append(row_info) # append the row
    
    # load the weights
    # run the crops through your neural network and print them out
    import pickle
    import string
    letters = np.array([_ for _ in string.ascii_uppercase[:26]] + [str(_) for _ in range(10)])
    params = pickle.load(open('q3_weights.pickle','rb'))
    ##########################
    ##### your code here #####
    ##########################
    for row_info in data:
        row_info = np.array(row_info)
        h1 = forward(row_info, params, 'layer1', sigmoid)
        probs = forward(h1, params, 'output', softmax)
        text_row = ''
        idx = np.argmax(probs[0, :])
        for i in range(probs.shape[0]):
            idx2 = np.argmax(probs[i, :])
            text_row += letters[idx2]
        print(text_row)