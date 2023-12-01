import numpy as np

import skimage
import skimage.measure
import skimage.color
import skimage.restoration
import skimage.filters
import skimage.morphology
import skimage.segmentation

# takes a color image
# returns a list of bounding boxes and black_and_white image
def findLetters(image):
    bboxes = []
    bw = None
    # insert processing in here
    # one idea estimate noise -> denoise -> greyscale -> threshold -> morphology -> label -> skip small boxes 
    # this can be 10 to 15 lines of code using skimage functions

    ##########################
    ##### your code here #####
    ##########################

    # Estimate noise
    # sigma_est = skimage.restoration.estimate_sigma(image, multichannel=True, average_sigmas=True)
    # Denoise
    image = skimage.restoration.denoise_bilateral(image, channel_axis=-1)
    # Greyscale
    image = skimage.color.rgb2gray(image)
    # Threshold
    thresh = skimage.filters.threshold_otsu(image)
    bw = image < thresh
    # Morphology
    bw = skimage.morphology.closing(bw, skimage.morphology.square(5))
    # Label
    label_im = skimage.morphology.label(bw, connectivity=2)
    label = skimage.measure.label(label_im)
    # Skip small boxes
    for i in skimage.measure.regionprops(label):
        if i.area >= 200:
            bboxes.append(i.bbox)

    bw = 1.0 - bw

    return bboxes, bw