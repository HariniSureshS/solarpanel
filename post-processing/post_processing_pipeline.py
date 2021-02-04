#!/usr/bin/env python
# coding: utf-8

# In[7]:

from solaris.data import data_dir
import solaris as sol
import os
import skimage
import geopandas as gpd
from matplotlib import pyplot as plt
from shapely.ops import cascaded_union
import cv2
import rasterio as rio
import pycocotools.mask as mask_util
import shapely
import math
import numpy as np
import json
from shapely.geometry import Polygon, Point
from osgeo import gdal
import csv
import pandas as pd

#!/usr/bin/env python
# coding: utf-8

# In[7]:

class Post_Process:
    def __init__(self, coco_file , tif_img, img_name, threshold, crs_value='EPSG:4326'):
        '''
        class takes model prediction (coco json format) c
        and reference image, for post processing
        Returns, binary mask and dataframe with area and oriantation
        '''
        self.coco_file = coco_file
        self.tif_img = tif_img
        self.geo_df = None
        self.bin_mask = None
        self.result_dict = []
        self.img_name = img_name
        self.threshold = threshold
        self.crs_value = crs_value
        
    
    def get_full_post_process(self):
        ''' Full post process mask
        Returns, Tuple (binary mask and dataFrame)'''
        self.bin_mask = self.get_bin_mask()
        self.geo_df = self.get_px_coords_df()
        #self.geo_df = self.get_px_coords()
        self.geo_df = self.get_geo_coords()
        self.geo_df = self.get_lat_long()
        self.geo_df = self.get_area()
        self.geo_df = self.get_azimuth()

       
    
        # Return the binary mask and the geo dataframe 
        return  self.bin_mask, self.geo_df
        
    
    def get_bin_mask(self):
        '''Decods the model prediction in Json format with the encoded masks
        Retruns the binary mask '''
        ## Read and parse our json file
        with open(self.coco_file, 'r') as my_file:
            data = my_file.readlines()
            # Parse file
            obj = json.loads(data[0])
        
        #poly_loc_list = []
        ## iterate and decode segmentations
        self.bin_mask = np.zeros(obj[0]['segmentation']['size'])

        for i in range(len(obj)):
            # Check the prediction score
            if obj[i]['score'] > self.threshold:
                seg_dict = {}
                seg_dict = {'size': obj[i]['segmentation']['size'], 'counts' : obj[i]['segmentation']['counts']}
        
                poly = mask_util.decode(seg_dict)[:, :]
            
                self.bin_mask += poly
            
        return  self.bin_mask
    
    def calc_azimuth(self, g):
        '''takes a geometry  and returns the angle'''        
        a = g.minimum_rotated_rectangle
        l = a.boundary
        coords = [c for c in l.coords]
        segments = [shapely.geometry.LineString([a, b]) for a, b in zip(coords,coords[1:])]
        longest_segment = max(segments, key=lambda x: x.length)

        p1, p2 = [c for c in longest_segment.coords]
        angle = math.degrees(math.atan2(p2[1]-p1[1], p2[0]-p1[0]))
        return angle
    
    def get_area(self):
        '''
        calculate area for our polygons and append to the geo_df
        '''
        # reproject to meter coordinates
        self.geo_df = self.geo_df.to_crs('EPSG:5703')
        # If it we have the df
        # Get the area and append to the df
        if self.geo_df is not None :
            # Calculate the area
            self.geo_df['area(square meter)'] = ((self.geo_df['geometry'].area)) # In sequare meter #/10.764
            # Back to original 
            self.geo_df = self.geo_df.to_crs(self.crs_value)
        else:
            print("No dataframe acquired, yet. Use get_geo_df")
        
        return self.geo_df
    
    
    def get_azimuth(self):
        '''Calculate the azimuth angle'''
        # Make sure we already have the geo_df calculated
        list_azimuth = []
        if self.geo_df is not None :
            for i in range(len(self.geo_df)):
                g = self.geo_df.iloc[i].geometry
                angle = self.calc_azimuth(g)
                list_azimuth.append(angle)
                
            self.geo_df['Roof_Azimuth'] = list_azimuth
        
        else: 
             print("No dataframe acquired, yet. Use get_geo_df")
        
        return self.geo_df
    
    def get_px_coords_df(self):
        '''
        Takes a tif image and returns a geo dataframe
        '''
        if self.bin_mask is not None:
            
            self.geo_df = sol.vector.mask.mask_to_poly_geojson(pred_arr=self.bin_mask, 
                                                 reference_im=self.tif_img,  
                                                 min_area=1, simplify=True) #min_area=1
            
            self.geo_df = self.geo_df.drop(['value'], axis = 1) 
            
            
        else:
            print("Claclulate binary mask first using get_bin_mask")
            
        return self.geo_df
    
    def get_px_coords(self):
        """Function to get geo coordinates given a pixel coordinates"""
        xs_list = []
        ys_list = []
        # Get thelist of pixel lat/long
       # First we get the pixel lat/long values
        self.geo_df['pixel_Center_point'] = self.geo_df['geometry'].centroid
        #Extract lat and lon from the centerpoint
        self.geo_df["pixel_x"] = self.geo_df.pixel_Center_point.map(lambda p: p.x)
        self.geo_df["pixel_y"] = self.geo_df.pixel_Center_point.map(lambda p: p.y)
              
        return self.geo_df
    
    def get_geo_coords(self):
        '''Convert the each pixel point 
        to a georefrenced point with lat/long coordinates'''
        # Now convert the pixel row/col to lat/long
        # unravel GDAL affine transform parameters
        coords = [list(poly.exterior.coords) for poly in self.geo_df.geometry]
        geo_poly = []
        geo_coords = []

        for poly in coords:
            geo_poly = []
            #print(poly)
            for points in poly:
                x = points[0]
                y = points[1]
        
                (px, py) = rio.transform.xy(self.tif_img.transform, y, x, offset='center')
                poly = (px, py)
                geo_poly.append(poly)
        
            geo_coords.append(geo_poly)
    
        geo_polys = []    
        for poly in geo_coords:
            geo_polys.append(Polygon(poly))
        
        # Add the new polygons to our gdf
        #self.geo_df['px_polygonss'] = self.geo_df['geometry'] # The old pixel polygons
        self.geo_df['geometry'] = geo_polys # assign the new polygons
        self.geo_df = self.geo_df.set_crs(self.crs_value) #'EPSG:4326'
        return self.geo_df
    
    def get_lat_long(self):
        """Function to get coordinates in latitude and longitude degress as a column in dataframe"""
        # reproject to meter coordinates
        self.geo_df = self.geo_df.to_crs('EPSG:5703')
        # Find the center of the polygons
        self.geo_df['points'] = self.geo_df['geometry'].centroid
        #Extract lat and lon from the centerpoint (This is extra)
        self.geo_df["latitude"] = self.geo_df.points.map(lambda p: p.x)
        self.geo_df["longitude"] = self.geo_df.points.map(lambda p: p.y)
        #self.geo_df = self.geo_df.drop(['center_point', 'polygons', 'pix_polygons', 'pixel_Center_point'], axis = 1)
        # Back to original crs
        self.geo_df = self.geo_df.to_crs(self.crs_value)
        return self.geo_df
    
    
    # Append the information to our json file and save it somewhere
    def append_to_json(self, filename):
        '''takes a ../filename.json and 
        save the dictionary with all the information
        in a JSON file on disk'''
        ## Read and parse our json file
        with open(self.coco_file, 'r') as my_file:
            data = my_file.readlines()
            # Parse file
            obj = json.loads(data[0])
        
        #poly_loc_list = []
        ## iterate and decode segmentations
        self.bin_mask = np.zeros(obj[0]['segmentation']['size'])
        
        for i in range(len(self.geo_df)):
            # Check the prediction score
            if obj[i]['score'] >= self.threshold:
                seg_dict = {}
                seg_dict = {'size': obj[i]['segmentation']['size'], 'counts' : obj[i]['segmentation']['counts']}
        
                poly = mask_util.decode(seg_dict)[:, :]
                seg_dict = {}
                seg_dict = {'size': obj[i]['segmentation']['size'], 'counts' : obj[i]['segmentation']['counts']}
                poly = mask_util.decode(seg_dict)[:, :]
                poly = poly.tolist()
                 # Create new Json file to append our information to (this will contains polygons with high scores only)
                new_json = {}
                new_json = {'image_name': self.img_name, 
                            'geometry': (np.asarray(self.geo_df.iloc[i].geometry.exterior.coords)).tolist(),
                            'longitude': self.geo_df.iloc[i].longitude, 
                            'latitude':self.geo_df.iloc[i].latitude, 
                            'area(square meter)': self.geo_df.iloc[i]['area(square meter)'],
                            'Roof_Azimuth':self.geo_df.iloc[i].Roof_Azimuth}
                
                self.result_dict.append(new_json)
                
        all_json_list = []
        with open(self.coco_file, 'r+') as my_file:
            data = my_file.readlines()
            # Parse file
            obj = json.loads(data[0])
        for i in range(len(self.geo_df)):
            if obj[i]['score'] >= self.threshold: 
                dict_copy = obj[i].copy()
                # Update the dictionary
                dict_copy.update(self.result_dict[i])
                all_json_list.append(dict_copy)
                
                with open(filename, 'w') as outfile:
                    json.dump(all_json_list, outfile)
                    
        
    # Helper method to save our list of dictionaries as a csv file
    def save_to_csv(self, file_name):
        '''Takes a file name with csv extension'''
        keys = self.result_dict[0].keys()
        with open(file_name, 'w', newline='')  as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows( self.result_dict)


