# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 11:51:13 2019
@author: Erik Neemann (erikneemann -at- gmail -dot- com)
https://github.com/eneemann

This script is used to perform object-based image classification on remotely
sensed imagery using a Random Forest Model.  It expects a 3-band RGB GeoTIFF
from a UAS or other similar source. Several parameters need to be updated at the
top of the script in order to function properly:
    - Image directory (image_dir)
    - Image filename (image_file)
    - Training data directory (train_dir)
    - Training data file (train_file) - Must be shapefile with "Class" field
    - Output shapefile name (shape_filename)
    - Number of trees in random forest model (numtrees, recommended between 100 and 500)
    - Land cover classes to use (classes)
    - Colors for plotting land cover classes (colors)

"""
import time
start_time = time.time()

###############################################################################
# Import most of the needed libraries up front
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import rasterio as rio
from rasterio import features
import arcpy
from arcpy import env

import fiona
from skimage.segmentation import quickshift
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float, img_as_uint
from skimage import exposure

###############################################################################
# Set variables to be used
# Directory where images are stored
image_dir = r'C:\Users\Erik\Desktop\MSGIS Spring 2019\6161 - Capstone\Phragmites\Data\PMG UAS'
image_file = 'HowardSlough092018_Orthomosaic_export_WedSep26171037.tif'      # Image to be classified
# Directory where training data is stored
train_dir = r'C:\Users\Erik\Desktop\MSGIS Spring 2019\6161 - Capstone\Phragmites\Data'
train_file = 'Phrag_Training_Data_Classes_UAS.shp'    # Land cover training polygons
# Name of output Phragmites boundary shapefile...it will be placed in train_dir
shape_filename = "UAS_Phrag_Boundary.shp"
numtrees = 100  # Number of trees in random forest model

# Specify classes and colors in final classified image
# These lists must have the same number of elements!
classes = ["Water", "Live Phragmites", "Algae", "Dead Phragmites",
           "Bare Ground", "Native Emergent", "No Data"]
colors = ['blue', 'forestgreen', 'aqua', 'saddlebrown', 'orange', 'limegreen', 'black']

###############################################################################
# Step 1: Load image and clean up the data
###############################################################################

# Set working directory where the data is located
os.chdir(image_dir)
env.overwriteOutput = True

# Open image file and read UAS bands
filename = os.path.join(image_dir, image_file)
with rio.open(filename) as src:
    red = src.read(1)
    green = src.read(2)
    blue = src.read(3)

# Save shape of image for later to use when re-shaping the array
original_shape = red.shape
    
# Create function to normalize array values
def normalize(array):
    """Normalizes numpy arrays into scale 0.0 - 1.0"""
    array_min, array_max = array.min(), array.max()
    return ((array - array_min)/(array_max - array_min))

print("time elapsed: {:.2f}s".format(time.time() - start_time))

# Normalize the individual bands
redn = normalize(red)
greenn = normalize(green)
bluen = normalize(blue)

###############################################################################
# Step 2: Calculate additional bands
###############################################################################

"""
This block is not applicate for RGB imagery

# Create function to calculate NDVI
def calc_ndvi(nir,red):
    '''Calculate NDVI from integer arrays'''
    nir = nir.astype('f4')
    red = red.astype('f4')
    ndvi = (nir - red) / (nir + red)
    return ndvi

# Create function to calculate NDWI
def calc_ndwi(nir,green):
    '''Calculate NDVI from integer arrays'''
    nir = nir.astype('f4')
    green = green.astype('f4')
    ndwi = (green - nir) / (green + nir)
    return ndwi

# Calculate NDVI and NDWI
ndvi = calc_ndvi(nirn, redn)
ndwi = calc_ndwi(nirn, greenn)

# Plot NDVI
plt.figure(figsize=(20,12))
plt.imshow(ndvi, cmap='BrBG')
plt.colorbar()
plt.title("NDVI Image")
plt.tight_layout()

# Plot NDWI
plt.figure(figsize=(20,12))
plt.imshow(ndwi, cmap='BrBG')
plt.colorbar()
plt.title("NDWI Image")
plt.tight_layout()
"""

# Create RGB color composite
rgb = np.dstack((redn, greenn, bluen))

# Rescale images to improve image contrast (2-98 percentile)
p2, p98 = np.percentile(rgb, (2, 98))
rgb_rescale = exposure.rescale_intensity(rgb, in_range=(p2, p98))

# Plot Red-Green-Blue Image (RGB)
plt.figure(figsize=(20,12))
plt.imshow(rgb_rescale)
plt.title("RGB UAS Image")
plt.tight_layout()
plt.xticks([])
plt.yticks([])
plt.show()

###############################################################################
# Step 3: Load training data as raster
###############################################################################

# Convert training shapefile to raster, matching grid of main image
# Set several environment variables to match raster of image to be used
snapRaster = os.path.join(image_dir, image_file)
env.snapRaster = snapRaster
env.extent = snapRaster
env.mask = snapRaster
env.cellSize = snapRaster
spatial_ref = arcpy.Describe(snapRaster).spatialReference
env.outputCoordinateSystem  = spatial_ref
os.chdir(train_dir)
section_time = time.time()
in_features = os.path.join(train_dir, train_file)
out_name = os.path.join(train_dir, 'UAS_train_classes.tif')
# Convert training data polygons to raster
arcpy.PolygonToRaster_conversion(in_features, 'Class', out_name, "", "", snapRaster)
print("Time elapsed for ArcPy PolygonToRaster: {:.2f}s".format(time.time() - section_time))

# Open training data raster and read classes
train_raster = os.path.join(train_dir, out_name)
with rio.open(train_raster) as train:
    train_classes = train.read(1)
    
# Plot training data raster   
plt.figure(figsize=(20,12))
# Change cmap and vmax if more than 9 classes
plt.imshow(train_classes, cmap='Set1', vmin=0, vmax=9)
plt.colorbar()
plt.title("Training Classes")
plt.tight_layout()
         
print("time elapsed: {:.2f}s".format(time.time() - start_time))

###############################################################################
# Step 4: Segment image into objects
###############################################################################

section_time = time.time()

# Create image variable as float data type for segmentation algorithm
image = img_as_float(rgb_rescale[:, :, :])

# Use scikit-learn quickshift algorithm to segment image into objects
segments_quick = quickshift(image, kernel_size=6, max_dist=10, ratio=0.75)

print('Quickshift number of segments: {}'.format(len(np.unique(segments_quick))))
#colors_quick = color.label2rgb(segments_quick, image, kind='avg')

print("time elapsed for quickshift: {:.2f}s".format(time.time() - section_time))

     
###############################################################################
#### Optional Step to output segment boundaries to shapefile 
###############################################################################
# This block outputs image segment boundaries as a shapefile
# ***This may take a long time and create a large shapefile***
#
## Convert segments to uint prior to exporting
#seg_quick_uint = img_as_uint(segments_quick)
#
#section_time = time.time()
#schema = {"geometry": "Polygon", "properties": {"value": "int"}}
#shape_filename = "UAS_Object_Boundaries.shp"
#
#with rio.open(filename) as src:
#    with fiona.open(shape_filename, "w", "ESRI Shapefile",
#                crs=src.crs.data, schema=schema) as out_file:
#        print("Closing shapefile w/o writing")
#        exit
#
#with rio.open(filename) as src:
#        for shape, value in features.shapes(seg_quick_uint, transform=src.transform):
#            records = [{"geometry": shape, "properties": {"value": int(value)}}]
#            with fiona.open(shape_filename, "a", "ESRI Shapefile",
#                            crs=src.crs.data, schema=schema) as out_file:
#                out_file.writerecords(records)
#                if len(out_file) % 20000 == 0:
#                    print("Current length of file is: {}".format(len(out_file)))
#
#print("time for shapefile: {:.2f}s".format(time.time() - section_time))

# Plot image with segment boundaries on top
plt.figure(figsize=(20,12))
plt.imshow(mark_boundaries(image, segments_quick))
plt.title("Quickshift Boundaries")
plt.tight_layout()
plt.show()
print("time shapefile plus graphic: {:.2f}s".format(time.time() - section_time))

###############################################################################
# Step 5: Calculate band-averages for each object
###############################################################################

section_time = time.time()
# Send data to ArcPy for zonal statistics (mean)
# Tool fails unless spatial extension is checked out, check out with this block of code
if arcpy.CheckExtension("spatial") == "Available":
    print("Spatial Analyst license is Available")
    arcpy.CheckOutExtension("spatial")
else:
    print("The Spatial Analyst license is Not Available")

# Create list of bands to be used in random forest model
bands_list = [bluen, greenn, redn]

# Get raster data for coordinate info to geolocate raster in arcpy
templateRaster = arcpy.sa.Raster(snapRaster)

# Lower left coordinate of block (in map units)
mx = templateRaster.extent.XMin
my = templateRaster.extent.YMin
sr = templateRaster.spatialReference

# Convert segments from numpy arrary to arcpy raster
# First, convert data type to int16 (or int32)
seg_quick_int = segments_quick.astype(np.int16)
segRaster = arcpy.NumPyArrayToRaster(seg_quick_int, arcpy.Point(mx, my),
                                     templateRaster.meanCellWidth,
                                     templateRaster.meanCellHeight)

# Create empty list, feed all bands into arcpy rasters
# then calculate mean within zone (image segment) and send raster
# back as a numpy array and append it to the list
# Block results in a list of arrays, each representive an object-averaged band
section_time = time.time()
allBandMeans = []
for band in bands_list:
    bandRaster = arcpy.NumPyArrayToRaster(band, arcpy.Point(mx, my),
                                          templateRaster.meanCellWidth,
                                          templateRaster.meanCellHeight)
    zoneMeans = arcpy.sa.ZonalStatistics(segRaster, "VALUE", bandRaster, "MEAN")
    bandMean = arcpy.RasterToNumPyArray(zoneMeans)
    allBandMeans.append(bandMean)
arcpy.CheckInExtension("spatial")

# Append training data to the list of object-averaged band arrays
allBandMeans.append(train_classes)
# Create image stack for classification
allBandsRavel = []  # create empty list that bands will be added to

# Loop through object-averaged band arrays and unravel each 2D array into 1D column vector
for band in allBandMeans:
    print(band.shape)
    raveled = band.ravel().reshape(-1,1)
    print(raveled.shape)
    allBandsRavel.append(raveled)
# Combine the 1D column vectors together into a Pandas dataframe
allBands_df = pd.DataFrame(np.hstack(allBandsRavel), columns=['blue', 'green', 'red', 'Class'])  
print("time to get zonal statistics and ravel bands: {:.2f}s".format(time.time() - section_time))

###############################################################################
# Step 6: Train random forest classification model
###############################################################################

section_time = time.time()
# Build training data set that will act on individual pixels (objects not needed)
# Add training classes to band list
bands_list.append(train_classes)
# Loop through image band arrays and unravel each 2D array into 1D column vector
allBandsRavel_p = []
for band in bands_list:
    print(band.shape)
    raveled = band.ravel().reshape(-1,1)
    print(raveled.shape)
    allBandsRavel_p.append(raveled)
# Combine the 1D column vectors together into a Pandas dataframe
all_bands_pixel = pd.DataFrame(np.hstack(allBandsRavel_p), columns=['blue', 'green', 'red', 'Class'])
# Trim data down to valid data within training polygons
# All data not belonging to training polygon has value of 15, remove the 15s
all_bands_pixel_train = all_bands_pixel[all_bands_pixel.Class != 15]

# Separate data to independent/dependent variables
# Indexes are image bands (ind. var.), 'Class' is training data (dep. var.)
indexes = ['blue', 'green', 'red']
x = all_bands_pixel_train.loc[:, indexes].values       # independent
y = all_bands_pixel_train.loc[:, ['Class']].values     # dependent

# Create variable that will be used for predicting object classes
x_objects = allBands_df.loc[:, indexes]

# Split data into training and validation (test) sets at 75/25%
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .25, random_state = 0)

# Fit Random Forest model to the dataset
section_time = time.time()
from sklearn.ensemble import RandomForestClassifier
trees = numtrees
# Set parameters of classifier (option to limit tree depth, if desired)
clf = RandomForestClassifier(n_estimators = trees, random_state = 0, max_features=3,
                             oob_score=True, verbose=1) #, max_depth=5)
clf.fit(x_train, y_train)   # Fit (train) the classifier model
print("time to run RF classifier ({} trees): {:.2f}s".format(trees, time.time() - section_time))

###############################################################################
# Step 7: Predict land cover class of each object
###############################################################################

# Predict data class of objects and reshape back into original shape
section_time = time.time()
allBands_df['Pred'] = clf.predict(x_objects)
print("time to predict: {:.2f}s".format(time.time() - section_time))
final_pred = allBands_df['Pred'].values.reshape(original_shape)

# Create dictionary of label colors and classes by zipping the lists
legend_labels = dict(zip(colors, classes))
# Create patches for legend
from matplotlib.patches import Patch
patches = [Patch(color=color, label=label) for color, label in legend_labels.items()]
# Plot final classified image with labels, save to PNG
plt.figure(figsize=(20,12))
plt.imshow(final_pred, cmap=matplotlib.colors.ListedColormap(colors), label=classes)
plt.title("UAS Image Final Classified")
plt.tight_layout()
plt.legend(handles=patches, facecolor="white", loc='best')
plt.xticks([])
plt.yticks([])
plt.savefig('UAS_Classified_Image.png')      # Export image as .png
plt.show()

###############################################################################
# Step 8: Export phragmites boundary to shapefile
###############################################################################

# Export phragmites boundary
# First, need to combine all non-Phragmites classes into one class
# Phrag = 1, so we will convert all other classes to 0 

# Copy the predicted class array
temp_arr = np.copy(final_pred)
# Convert all non-Phragmites classes (not a 1) to 0
temp_arr[temp_arr != 1] = 0
# Convert to uint before exporting data
pred_uint = img_as_uint(temp_arr)       # This line alters the values and converts 1 to 65535
# Change Phrag values back to 1
pred_uint[pred_uint == 65535] = 1       # set phrag values (temporarily 65535, back to 1)

section_time = time.time()

# Set simple schema to write out boundary shapefile, use the CRS from the image (UTM 12N)
# First create the shapefile without writing anything
schema = {"geometry": "Polygon", "properties": {"value": "int"}}
with rio.open(filename) as src:
    with fiona.open(shape_filename, "w", "ESRI Shapefile",
                crs=src.crs.data, schema=schema) as out_file:
        print("Closing shapefile w/o writing")
        exit
# Then start writing data to the shapefile
with rio.open(filename) as src:
        for shape, value in features.shapes(pred_uint, transform=src.transform):
            records = [{"geometry": shape, "properties": {"value": int(value)}}]
            with fiona.open(shape_filename, "a", "ESRI Shapefile",
                            crs=src.crs.data, schema=schema) as out_file:
                out_file.writerecords(records)
                if len(out_file) % 1000 == 0:
                    print("Current length of file is: {}".format(len(out_file)))
print("time for shapefile: {:.2f}s".format(time.time() - section_time))

###############################################################################
# Calculate and print metrics and confusion matrix
###############################################################################

# Output several metrics calculated from sklearn module
section_time = time.time()
print("Test score is: {}".format(clf.score(x_test, y_test)))
print("Train score is: {}".format(clf.score(x_train, y_train)))
print("OOB score is: {}".format(clf.oob_score_))

# Output confusion matrix for full data set, test/validation data, and training data 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
y_pred = clf.predict(x)
y_test_pred = clf.predict(x_test)
y_train_pred = clf.predict(x_train)
print('Confusion matrix and Kappa score on "full" training data:')
print(confusion_matrix(y, y_pred))
print(cohen_kappa_score(y, y_pred))
print('Confusion matrix and Kappa score on test data subset:')
print(confusion_matrix(y_test, y_test_pred))
print(cohen_kappa_score(y_test, y_test_pred))
print('Confusion matrix and Kappa score on train data subset:')
print(confusion_matrix(y_train, y_train_pred))
print(cohen_kappa_score(y_train, y_train_pred))

print("Time elapsed for metrics: {:.2f}s".format(time.time() - section_time))

print("total time elapsed: {:.2f}s".format(time.time() - start_time))

###############################################################################
# Calculate and plot variable importances
###############################################################################

# Look at variable importances to see which bands were most important to classifier
importances = clf.feature_importances_
std = np.std([clf.feature_importances_ for tree in clf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
# Print the feature ranking
print("Feature ranking:")
for f in range(x.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure(figsize=(10,8))
plt.title("Feature Importances")
plt.bar(range(x.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(x.shape[1]), indices)
plt.xlim([-1, x.shape[1]])
plt.show()

print("Feature Importances:")
print(clf.feature_importances_)

print("Pixel counts by class:")
unique, counts = np.unique(final_pred, return_counts=True)
dict(zip(unique, counts))