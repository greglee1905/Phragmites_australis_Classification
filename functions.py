import rasterio as rio
import numpy as np

#Function to open and return an array for the image without the null values
def open_image(path_to_image):
    with rio.open(path_to_image, 'r+') as src:
        test_image = src.read()
        test_image_meta = src.meta

        #Remove the null portion of the image: 6
        test_image = test_image[:5][:][:]

    return test_image

#Image Preperation Function
def image_prep(im):
    rolled = np.rollaxis(im,0,3)

    #This will break down the tiles into 5x32x32 except for the edges....
    tiles = [rolled[x:x+32,y:y+32,:5] for x in range(0,rolled.shape[0],32) for y in range(0,rolled.shape[1],32)]

    #Going to zero fill the border images to create even sized images
    indices = []
    filled = tiles

    for index,arrays in enumerate(tiles):

        #Fill tiles which are not of shape 32x32x5
        if arrays.shape != (32,32,5):
            indices.append(index)
            test = np.zeros((32,32,5))
            test[:arrays.shape[0],:arrays.shape[1]] = arrays
            filled[index] = test

    return np.stack(filled)

#Deconstructing neural network output back to image
def unstack(prediction):
    #Unstack the image first
    *image_unstack, = prediction

    image_mask  = np.empty((0,320,1))
    count = 0
    temp = np.empty((32,0,1))

    #Convert back to a matrix
    for things in image_unstack:
        count +=1
        temp = np.append(temp,things,axis =1)
        if count == 10:
            image_mask = np.append(image_mask,temp,axis=0)
            count = 0
            temp = np.empty((32,0,1))

    #Unroll the image back to original axis shapes
    return (np.rollaxis(image_mask,2,0))

#This takes an input mask of (1,X,Y) and transforms it into a mask
def reshaper(image_mask):
    image_mask = image_mask.reshape(image_mask.shape[1],image_mask.shape[2])
    image_mask[image_mask>1] = 0
    return (np.ma.masked_where(image_mask == 0, image_mask))
