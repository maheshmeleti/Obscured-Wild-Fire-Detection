import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from moviepy.video.io.VideoFileClip import VideoFileClip
from tqdm import tqdm

def draw_boundaries(rgb_image, thresh_image, color=(0, 0, 255)):
    rgb_image = rgb_image.copy()
    contours, _ = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        cv2.drawContours(rgb_image, [contour], 0, color, 1)
    return rgb_image

def show_img(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def make_edges_zero(img):
    height, width = img.shape

    # Create a mask that selects the edge pixels
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[0,:] = 254
    mask[:,0] = 254
    mask[height-1,:] = 254
    mask[:,width-1] = 254

    # Set the edge pixels to zero
    img[mask==254] = 0
    
    return img

def fill(image):
    image = make_edges_zero(image)
    binary_image = image.astype("uint8")
    im_floodfill = binary_image.copy()
    h,w = binary_image.shape[:2]
    mask =np.zeros((h+2,w+2),np.uint8)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_out = binary_image | im_floodfill_inv
    return im_out


class bwareaopen():
    '''
    Call:
        remove_small = bwareaopen(image,area=300).remove_small()
        remove_large = bwareaopen(image,area=2000).remove_large()

    Input args:
        image= binary input image
        area = remove blobs having area less than or greater than this area

    methods:
        remove_small: remove blobs having area less than defined area
        remove_large: remove blobs having area greater than defined area

    return: processed binary image
    '''

    def __init__(self, image, area=0):
        '''
        Constructor
        '''
        self.area = area
        self.image = image

    def get_ccl(self):
        stats = cv2.connectedComponentsWithStats(self.image.astype("uint8"),
                                                 connectivity=8)
        self.nb_components, self.labels, self.stats, self.centroids = \
            stats[0], stats[1], stats[2], stats[3]

    def remove_small(self):
        bwareaopen.get_ccl(self)
        sizes = self.stats[1:, -1];
        nb_components = self.nb_components - 1
        img2 = np.zeros((self.labels.shape), dtype="uint8")
        for i in range(0, nb_components):
            if sizes[i] >= self.area:
                img2[self.labels == i + 1] = 255

        return img2

    def remove_large(self):
        bwareaopen.get_ccl(self)
        sizes = self.stats[1:, -1];
        nb_components = self.nb_components - 1
        img2 = np.zeros((self.labels.shape), dtype="uint8")
        for i in range(0, nb_components):
            if sizes[i] <= self.area:
                img2[self.labels == i + 1] = 255

        return img2
    

def put_text(img, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (0, 255, 0)
    thickness = 2
    # Get the size of the text
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = int((img.shape[1] - text_size[0]) / 2)
    text_y = 20
#     text_y = int((img.shape[0] + text_size[1]) / 2)
    
    cv2.putText(img, text, (text_x, text_y), font, font_scale, color, thickness)
    return img

def IR2Label(ir_image, thresh=180):
    ir_image = ir_image.copy()
    frame_gray = cv2.cvtColor(ir_image, cv2.COLOR_BGR2GRAY)
    ret, frame_thresholded = cv2.threshold(frame_gray,thresh,255,cv2.THRESH_BINARY)
    
    # dilate
    kernel = np.ones((5, 5), np.uint8)
    img_dilation = cv2.dilate(frame_thresholded, kernel, iterations=2)
    # fill
    image_filled = fill(img_dilation)
    #erode
    img_erosion = cv2.erode(image_filled, kernel, iterations=1)
    return img_erosion

def IR2LabelSmooth(ir_image):
    kernel = np.ones((5,5),np.float32)/25
    frame_ir_smooth = cv2.filter2D(ir_image,-1,kernel)
    IR_label = IR2Label(frame_ir_smooth, thresh = 100)
    IR_label_small_removed = bwareaopen(IR_label, 200).remove_small()
    return IR_label_small_removed