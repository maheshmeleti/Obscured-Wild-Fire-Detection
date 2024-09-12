import cv2
import numpy as np

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

    def __init__(self,image,area=0):
        '''
        Constructor
        '''
        self.area = area        
        self.image = image
    
    def get_ccl(self):                
        stats = cv2.connectedComponentsWithStats(self.image.astype("uint8"),
                                                 connectivity=8)
        self.nb_components, self.labels, self.stats, self.centroids =\
        stats[0],stats[1],stats[2],stats[3]           
    
    def remove_small(self):
        bwareaopen.get_ccl(self)            
        sizes = self.stats[1:, -1]; nb_components = self.nb_components - 1
        img2 = np.zeros((self.labels.shape),dtype="uint8")        
        for i in range(0, nb_components):
            if sizes[i] >= self.area:
                img2[self.labels == i + 1] = 255
        
        return img2
    
    def remove_large(self):
        bwareaopen.get_ccl(self)            
        sizes = self.stats[1:, -1]; nb_components = self.nb_components - 1
        img2 = np.zeros((self.labels.shape),dtype="uint8")        
        for i in range(0, nb_components):
            if sizes[i] <= self.area:
                img2[self.labels == i + 1] = 255            
        
        return img2    

class bwboundaries():
    '''
    Call: 
        boundaries = bwboundaries(image=image,prop='no_holes').extract()        
    
    Input args:
        image= binary input image 
        prop:
            'no_holes': exclude holes for boundary extraction
            'include_holes':include holes for boundary extraction
                
    methods:
        extract: extract boundaries from labeled blobs        
    
    return: list containing boundary coordinates for each blob
    '''

    def __init__(self, **kwargs):
        '''
        Constructor
        '''
        self.__dict__.update(kwargs)
        
    def extract(self):
        
        image = self.image
        prop = self.prop
        
        if prop == 'include_holes': method = cv2.RETR_LIST
        else: method = cv2.RETR_EXTERNAL
                
        _,contours,_ = cv2.findContours(image.copy(),method, cv2.CHAIN_APPROX_NONE)  
        
        rg = regionprops(image=image,properties=['Centroid']).get_properties()    
        new_bounds = [None]*len(rg['Centroid'])            
        label = rg['LabeledImage']
        
        for i in range(len(contours)):            
            cnt = np.reshape(contours[i],(len(contours[i]),2))        
            m_label = (label[cnt[0,1],cnt[0,0]])        
            new_bounds[np.int(m_label)-1] = cnt
        
        new_bounds = [element for element in new_bounds if element is not None]
        return new_bounds

class color_space_conversion():
    
    '''
    Call: 
        gray = color_space_conversion(image = im,method='rgb2gray').convert()
        hsv = color_space_conversion(image = im,method='rgb2hsv').convert()        
        bgr = color_space_conversion(image = hsv,method='hsv2bgr').convert()
        
    Input args:
        image= rgb/hsv color input image
        method: 
              'rgb2gray' : convert color image to gray image
              'rgb2hsv'  : convert color image to hsv image
              'hsv2rgb'  : convert color image to rgb image      
                
    methods:
        convert: convert as per defined color space        
    
    return: out image with defined color space
    '''
    
    def __init__(self,**kwargs):
        
        self.__dict__.update(kwargs)
    
    
    def rgb2gray(self):
        
        image = self.image
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        self.gray = gray
    
    def rgb2hsv(self):
        
        image = self.image
        hsv = cv2.cvtColor(image.astype("float32"),cv2.COLOR_BGR2HSV)
        hsv[:,:,0]/=360
        hsv[:,:,2]/=255
        self.hsv = hsv
    
    def hsv2rgb(self):
        image = self.image
        image[:,:,0]*=360
        image[:,:,2]*=255
        
        rgb = cv2.cvtColor(image,cv2.COLOR_HSV2BGR)
        self.rgb = rgb
        
    def convert(self):
        
        method = self.method
        
        if method == 'rgb2gray':
            color_space_conversion.rgb2gray(self)
            return self.gray
        
        if method == 'rgb2hsv':
            color_space_conversion.rgb2hsv(self)
            return self.hsv
        
        if method == 'hsv2rgb':
            color_space_conversion.hsv2rgb(self)
            return np.uint8(self.rgb)

class im2bw():
    '''
    Call:
        image,th = im2bw(image=im,threshold='otsu').make_binary()
        image,th = im2bw(image=im,threshold=200).make_binary()
    
    Input args:
        image=gray or color image (if color image provided; it will be first converted into gray)
        threshold:
            'otsu': calculate otsu threshold over image and convert it into binary
            'user': it will be a user defined scaler by which image will be thresholded
    
    methods:
        make_binary: convert image into binary 
    
    return:
        image = thresholded image. This image will be in range of 0,255 unlike MATLAB which have 0,1
        threshold = threshold used by algorithm to make binary image (helpful in case of otsu)
    '''


    def __init__(self, **kwargs):
        '''
        Constructor
        '''
        self.__dict__.update(**kwargs)
        
    def make_binary(self):        
        
        image = self.image
        
        if len(image.shape)>2:
            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)            
        
        if self.threshold=='otsu':
            th,image = cv2.threshold(image,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)                        
            return image,th
        else:
            image = image>self.threshold
            return 255*image.astype("uint8"),self.threshold

class imfill():
    '''
    Call: 
        image = imfill(image=image).fill()
        
    Input args:
        image = binary input image
    
    method:
        fill: when called; it will fill image holes
    
    return:
        image: binary image with filled image holes
     
    '''
    def __init__(self, **kwargs):
        '''
        Constructor
        '''
        self.__dict__.update(kwargs)
    
    def fill(self):
        binary_image = self.image.astype("uint8")            
        im_floodfill = binary_image.copy()
        h,w = binary_image.shape[:2]
        mask =np.zeros((h+2,w+2),np.uint8)
        cv2.floodFill(im_floodfill, mask, (0,0), 255);
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)
        im_out = binary_image | im_floodfill_inv
        return im_out

class imfilter():
    '''
    Call:
        im = imfilter(image = im,kernel='gaussian',ksize = 15,sigma=7.5,padding='replicate').conv()
        im = imfilter(image = im,kernel='sobel_horizontal',padding='replicate').conv()
        im = imfilter(image = im,kernel='average',ksize=15,padding='replicate').conv()
        im = imfilter(image = im,kernel='laplacian',kernel_val = gauss,padding='replicate').conv()
        im = imfilter(image = im,kernel='user',kernel_val = gauss,padding='replicate').conv()
        
    Input args:
        image: input image color or gray        
        
        kernel:        
            'gaussian': create a gaussian kernel with defined size and variance
            'sobel_horizontal'/'sobel_vertical': create 3X3 size vertical and horizontal sobel kernels
            'average': create box filter/ average filter of defined size
            'laplacian': create a 3X3 laplacian kernel
            'user': takes user defined kernel at the input
        
        kernel_val: if 'kernel' is 'user' then 'kernel_val' needs to be defined
        
        ksize: 'gaussian' and 'average' kernel will require user defined kernel size with a scaler value
        
        sigma: in case of 'gaussian' sigma needs to be defined with a scaler value             
        
        padding: 
            'replicate': padd input image with replication of boundary values
            'symmetric': padd input image with reflection of boundary values
    
    Methods:
        conv: perform convolution on input image
        
    return:
        filtered image with the same class type as input. 
        In case of user defined filter, type cast to float32 is recommended 
            
    '''

    def __init__(self, **kwargs):
        '''
        Constructor
        '''
        self.__dict__.update(kwargs)
        kernel_type = self.kernel
        
        if kernel_type == 'user':
            self.kernel=self.kernel_val                
        else:
            func = 'imfilter.'+kernel_type.lower()+'(self)'
            self.kernel = eval(func)    
    
    def conv(self):
        
        image = self.image
        kernel = self.kernel
        padding = self.padding
                    
        if padding=='replicate':method = cv2.BORDER_REPLICATE
        elif padding=='symmetric':method = cv2.BORDER_REFLECT
                
        filter_im = cv2.filter2D(image,-1,kernel,borderType=method)
        return filter_im   
    
    def gaussian(self):
        ksize= self.ksize
        sigma = self.sigma
        gauss = cv2.getGaussianKernel(ksize,sigma)
        gauss = gauss.T*gauss
        return gauss
    
    def average(self):
        mask = self.ksize
        kernel = np.ones((mask,mask),dtype="float32")/(mask*mask)   
        return kernel
    
    def sobel_vertical(self):
        kernel = np.array([1,     0,    -1,
                  2,     0,    -2,
                  1,     0,    -1])
        
        return kernel.reshape((3,3))        
    
    def sobel_horizontal(self):
        kernel = np.array([1,     2,     1,
                           0,     0,     0,
                           -1,    -2,    -1])
        
        return kernel.reshape((3,3)) 
    
    def laplacian(self):
        kernel = np.array([0.1667,    0.6667,    0.1667,
                           0.6667,   -3.3333,    0.6667,
                           0.1667,    0.6667,    0.1667])
        return kernel.reshape((3,3))

class logical_operations():
    '''
    Call:
        or_im = logical_operations(image_1=remove_large,image_2=remove_small).bitwise_or()
        and_im = logical_operations(image_1=remove_large,image_2=remove_small).bitwise_and()
        not_im = logical_operations(image_1=image).bitwise_not()
        xor_im = logical_operations(image_1=remove_large,image_2=remove_small).bitwise_xor()
        
    Input args:
        image_1,image_2 = both images two perform 'and','or' and 'xor' operations.
                          in case of 'not' operation, only 'image_1' will be required.
    
    Methods:
        bitwise_or: to perform logical 'or' operation between two images
        bitwise_and: to perform logical 'and' operation between two images
        bitwise_not: to perform logical 'not'/'complement' operation on input image
        bitwise_xor: to perform logical 'xor' operation between two images
    
    return:
        image: processed image with "uint8" type 
        
    '''
    def __init__(self, **kwargs):
        '''
        Constructor
        '''
        self.__dict__.update(kwargs)        
        
    def bitwise_or(self):
        self.image_1 = self.image_1.astype("uint8")
        self.image_2 = self.image_2.astype("uint8")
        return cv2.bitwise_or(self.image_1,self.image_2).astype("uint8")
    
    def bitwise_and(self):
        self.image_1 = self.image_1.astype("uint8")
        self.image_2 = self.image_2.astype("uint8")
        return cv2.bitwise_and(self.image_1,self.image_2).astype("uint8")
    
    def bitwise_not(self):
        self.image_1 = self.image_1.astype("uint8")        
        return cv2.bitwise_not(self.image_1).astype("uint8")
    
    def bitwise_xor(self):
        self.image_1 = self.image_1.astype("uint8")
        self.image_2 = self.image_2.astype("uint8")
        return cv2.bitwise_xor(self.image_1,self.image_2).astype("uint8")


class regionprops():
    
    '''
    Call:
        rg = regionprops(image=image,properties=['BoundingBox','PixelIdxList',
                                             'Area','Centroid','ellipsefit']).get_properties()
    
    Input args:
        image: binary input image
        properties:
            'BoundingBox': return bounding rectangle
            'PixelIdxList': returns pixel index list as well as sub index list
            'Area': returns area of each blob
            'Centroid': returns centroid of each blob
            'ellipsefit': returns following properties if an ellipse
                'MajorAxisLength' of the blobs
                'MinorAxisLength' of the blobs
                'Eccentricity' of the blobs
                'Orientation' of the blobs
                'Diameter' of the blobs
            
    Methods:
        get_properties: Will extract defined properties of the blobs
        get_ccl: will return labeled image
    
    returns:
        'regionprops' returns all the defined properties in a dictionary format.
        As 'PixelIdxList' might not work similar to MATLAB; a list of 'PixelSubIdxList' also returned with it.
        'regionprops' return labeled image by default, regard less any property have been called.
        
        you can access dictionary in following way.
        
        area = rg['Area']
        bb = rg['BoundingBox']
        pix_idx = rg['PixelIdxList']
        pix_sub_idx = rg['PixelSubIdxList'] (This will be default whenever 'PixelIdxList' is called)
        MajorAxisLength = rg['MajorAxisLength']
        MinorAxisLength = rg['MinorAxisLength']
        Eccentricity = rg['Eccentricity']
        Orientation = rg['Orientation']
        diam = rg['Diameter']
        labeled image = rg['LabeledImage']        
    '''
    
    def __init__(self,**kwargs):
        '''
        Constructor
        '''                                
        self.__dict__.update(kwargs)        
        self.props = {}                                       
    
    def get_ccl(self):                
        stats = cv2.connectedComponentsWithStats(self.image.astype("uint8"),
                                                 connectivity=8)
        self.nb_components, self.labels, self.stats, self.centroids =\
        stats[0],stats[1],stats[2],stats[3]   
        self.props['LabeledImage'] = self.labels                 
                 
    def boundingbox(self):
        self.props['BoundingBox'] = self.stats[1:,:4]
        
    
    def pixelidxlist(self):
        indexes = []
        ids = []
                
        for i in range(self.nb_components-1):            
            idx = np.where(self.labels==i+1)            
            index = np.zeros((len(idx[0]),2),dtype="int32")            
            index[:len(index),0] = idx[0]
            index[:len(index),1] = idx[1]
            indexes.append(index)            
            id_xy = idx[0]*(self.labels.shape[1])+ idx[1]
            ids.append(id_xy)
            
        self.props['PixelSubIdxList'] = indexes
        self.props['PixelIdxList'] = ids        
    
    def fit_ellipse(self,cent,list_id):
        xbar,ybar = cent[1],cent[0] 
        x = list_id[:,0]-xbar
        y = -(list_id[:,1]-ybar)
        
        N = len(x)    
        uxx = np.sum(x**2)/N +1/12
        uyy = np.sum(y**2)/N +1/12
        
        uxy = np.sum(x*y)/N
        common = np.sqrt((uxx-uyy)**2+4*uxy**2)
        
        MajorAxisLength = 2*np.sqrt(2)*np.sqrt(uxx + uyy + common)
        MinorAxisLength = 2*np.sqrt(2)*np.sqrt(uxx + uyy - common)    
        Eccentricity = 2*np.sqrt((MajorAxisLength/2)**2 - (MinorAxisLength/2)**2) / MajorAxisLength
        Orientation = np.arctan(MinorAxisLength/MajorAxisLength)*180/np.pi
        
        return MajorAxisLength,MinorAxisLength,Eccentricity,Orientation
    
    def ellipsefit(self):
        
        if 'PixelSubIdxList' and 'Centroid' in self.props:
            idx_list = self.props['PixelSubIdxList']
            centroid = self.props['Centroid']
        else:
            regionprops.pixelidxlist(self)
            regionprops.centroid(self)
            idx_list = self.props['PixelSubIdxList']
            centroid = self.props['Centroid']
                
        ellipse_major = []
        ellipse_minor = []
        ellipse_angle = []
        ellipse_ecc = []
        
        for idx,cen in zip(idx_list,centroid):            
            ellipse = regionprops.fit_ellipse(self, cen, idx)
            ellipse_major.append(ellipse[0])
            ellipse_minor.append(ellipse[1])
            ellipse_ecc.append(ellipse[2])
            ellipse_angle.append(ellipse[3])
        
        self.props['MajorAxisLength'] = ellipse_major
        self.props['MinorAxisLength'] = ellipse_minor
        self.props['Orientation'] = ellipse_angle
        self.props['Eccentricity'] = ellipse_ecc
        self.props['Diameter'] = np.mean([ellipse_major,ellipse_minor],axis=0)            
    
    def area(self):
        self.props['Area'] = self.stats[1:,-1]
        
    def centroid(self):
        self.props['Centroid'] = self.centroids[1:]            
            
    def get_properties(self):
        
        if len(self.image.shape)>2:
            print("Please provide a binary image as input")
        else:
            regionprops.get_ccl(self)        
            props = self.properties                    
            for prop in props:
                func = 'regionprops.'+prop.lower()+'(self)'
                eval(func)            
            
            return self.props

class bwlabel():
    '''
    Call: 
        rbclabels = bwlabel(image = imrbc).label();
    
    Input:
        image: input image
    
    Method:
        label(): generates labeled image
    
    return:
        labeled image as output
    '''
    
    def __init__(self,**kwargs):
        self.__dict__.update(kwargs)
    
    def label(self):
        stats = cv2.connectedComponentsWithStats(self.image.astype("uint8"),
                                                 connectivity=8)
        return stats[1]

class imadjust():
    '''
    Call:
        out_im = imadjust(image=im).adjust()        
    
    Input:
        image: color or gray image at the input
    
    Methods:        
        adjust: adjust input image's contrast using inbuilt 'stretchlim' function
    
    return:
        out_im: adjusted image
    '''


    def __init__(self, **kwargs):
        
        self.__dict__.update(kwargs)
        if not 'lims' in kwargs.keys():
            self.lims = None
        
    def stretchlim(self):
        img = self.image
        
        tol_low = 0.01
        tol_high = 0.99
        
        if len(img.shape)>2:
            p = img.shape[2]
        else:
            if len(img.shape) == 1:
                img = np.reshape(img,(img.shape[0], 1, 1))
            else:
                img = np.reshape(img,(img.shape[0],img.shape[1],1))
            p = img.shape[2]
        
        nbins = 256
        ilowhigh = np.zeros((2,p),dtype="float32")
        for i in range(0,p):# Find limits, one plane at a time
            N = cv2.calcHist([img[:,:,i]],[0],None,[nbins],[0,nbins]);
            cdf = np.cumsum(N)/np.sum(N); #cumulative distribution function
            ilow = np.where(cdf > tol_low);
            ihigh = np.where(cdf >= tol_high);
            if ilow[0][0] == ihigh[0][0]:   # this could happen if img is flat
                ilowhigh[0,i] = 0
                ilowhigh[1,i] = nbins-1
            else:
                ilowhigh[0,i] = ilow[0][0]
                ilowhigh[1,i] = ihigh[0][0]
                            
        lowhigh = (ilowhigh)/(nbins-1);  # convert to range [0 1]
        self.lims = lowhigh
        return lowhigh
        
    def adjust(self):
        
        img = self.image
        lims = self.lims            
        
        if len(img.shape)>2:
            d = img.shape[2]
        else:
            img = np.reshape(img,(img.shape[0],img.shape[1],1))
            d = img.shape[2]
        
        out_image = np.zeros(img.shape,dtype="uint8")
        
        if lims is None:
            lims = imadjust.stretchlim(self)
        
        lIn,hIn = lims[0,:],lims[1,:]
        
        for p in range(d):                                
            lut = np.linspace(0,1,256)            
            lut =  np.maximum(lIn[p], np.minimum(hIn[p],lut));
            out_lut = np.uint8(np.round(255*(lut - lIn[p]) / (hIn[p] - lIn[p])));
            out_image[:,:,p] = cv2.LUT(img[:,:,p],out_lut) 
        
        return out_image
         

class strel():
    
    '''
        Input: 
            'size' = Disk Radius, Square Size        
        
        Return: Disk structural element
    '''
    
    def __init__(self,**kwargs):
        self.__dict__.update(kwargs)
        
    def disk(self):
        
        radius = self.size
        if radius<3:        
            y,x = np.ogrid[-radius: radius+1, -radius: radius+1]
            mask = x**2+y**2 
            mask = mask.max()-mask
            mask = mask>=radius**2
            mask = mask.astype("uint8")
            
        else:    
            ksize = 2*radius-1
            v = np.zeros((ksize,ksize),dtype="uint8");
            
            anchor = np.uint16(ksize/2)
            v[anchor,anchor] = 1
            
            theta = (np.pi/8)
            k = 2*radius/((np.cos(theta)/np.sin(theta))+1/np.sin(theta))
            
            dcomp = [1,1.41,1,1.41]
            rp = 2*np.uint16(np.floor(k/dcomp))+1
            
            d1 = np.ones((1,rp[0]),dtype="uint8")
            d2 = np.eye(rp[1],dtype="uint8")
            d3 = np.ones((rp[0],1),dtype="uint8")
            d4 = np.rot90(d2)
            
            h1 = cv2.dilate(v,d1);
            h2 = cv2.dilate(h1,d2);
            h3 = cv2.dilate(h2,d3);
            h4 = cv2.dilate(h3,d4);
            
            ids = np.where(h4>0)
            h4_crop = h4[np.min(ids[0]):np.max(ids[0])+1,np.min(ids[1]):np.max(ids[1])+1]
            
            ids = np.where(h4_crop>0)
            rd = ids[0]
            M = h4_crop.shape[0]
            rd -= np.floor((M)/2).astype("uint16")
            
            max_horiz_radius = max(rd);
            radial_difference = radius - max_horiz_radius;
            length = 2*(radial_difference-1) + 1;
            
            d5 = np.ones((length,1),dtype="uint8")
            d6 = np.ones((1,length),dtype="uint8")
            
            h5 = cv2.dilate(h4,d5);
            mask = cv2.dilate(h5,d6);
            
        return mask
    
    def square(self):
        
        mask = cv2.getStructuringElement(cv2.MORPH_RECT, (self.size, self.size))
        
        return mask
    
    def ellipse(self):

        mask=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.size, self.size))
        
        return mask


class Morphological_operations(object):
    '''
    Call:
        im_dilate = Morphological_operations(image=im_bw,kernel=strel(5).disk).imdilate()
        im_erode = Morphological_operations(image=im_bw,kernel=strel(5).disk).imerode()
        im_open = Morphological_operations(image=im_bw,kernel=strel(5).disk).imopen()
        im_close = Morphological_operations(image=im_bw,kernel=strel(5).disk).imclose()
        
    Input args:
        image: binary input image
        kernel: structuring element
    
    Methods:
        imdilate(): to perform morphological dilation
        imerode(): to perform morphological erosion
        imopen(): to perform morphological opening
        imclose(): to perform morphological closing
    
    return:
        image: processed image with "uint8" type 
        
    '''
    
    def __init__(self,**kwargs):
        '''
        Constructor
        '''
        self.__dict__.update(kwargs)             
    
    def imdilate(self,itern=1):

        dilation = cv2.dilate(self.image, self.kernel, itern)
        
        return dilation
    
    def imerode(self, itern=1):
        
        erosion = cv2.erode(self.image, self.kernel, itern)
        
        return erosion
    
    def imopen(self):
        
        opn=cv2.morphologyEx(self.image, cv2.MORPH_OPEN, self.kernel)
        
        return opn
        
    def imclose(self):
        
        closing = cv2.morphologyEx(self.image, cv2.MORPH_CLOSE, self.kernel)
        
        return closing
