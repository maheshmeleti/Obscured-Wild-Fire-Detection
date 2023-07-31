import numpy as np
import cv2

def make_edges_zero(img):
    height, width = img.shape

    # Create a mask that selects the edge pixels
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[0, :] = 254
    mask[:, 0] = 254
    mask[height - 1, :] = 254
    mask[:, width - 1] = 254

    # Set the edge pixels to zero
    img[mask == 254] = 0

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