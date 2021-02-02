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
from shapely.geometry import Polygon
from osgeo import gdal
import csv
import pandas as pd

class Post_Process:
    def __init__(self, coco_file , tif_img, img_name, threshold):
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
        
    
    def get_full_post_process(self):
        '''Full post process mask
        Returns, Tuple (binary mask and dataFrame) if result_to_dict is False
        Returns a Tuple (binary mask and dictionary structure data) if result_dict is True'''
        self.bin_mask = self.get_bin_mask()
        self.geo_df = self.__get_px_coords_df()
        self.geo_df = self.__get_px_coords()
        self.geo_df = self.__get_geo_coords()
        self.geo_df = self.__get_area()
        self.geo_df = self.__get_azimuth()
        # Reprojecting before extracting the lat/long on the map
        self.geo_df = self.__apply_projection()
        self.geo_df = self.__get_lat_long()
       
    
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
        
        
        
    
    def __get_px_coords_df(self, output_path=None):
        '''
        Takes a tif image and returns a geo dataframe
        '''
        if self.bin_mask is not None:
            
            if output_path:
                sol.vector.mask.mask_to_poly_geojson(pred_arr=self.bin_mask, 
                                                 reference_im=self.tif_img,
                                                 output_path=output_path,
                                                 output_type='geojson', simplify=True)
            
                self.gdf= gpd.read_file(output_path)
            # datagrame was not saved    
            else:
                geoms = sol.vector.mask.mask_to_poly_geojson(pred_arr=self.bin_mask, 
                                                 reference_im=self.tif_img,  
                                                 output_type='geojson', simplify=True)
                self.geo_df = gpd.GeoDataFrame(geoms)     
            
            self.geo_df = self.geo_df.drop(['value'], axis = 1) 
            
        else:
            print("Claclulate binary mask first using get_bin_mask")
            
        return self.geo_df
    
    

        
    
    def __calc_azimuth(self, g):
        '''takes a geometry  and returns the angle'''        
        a = g.minimum_rotated_rectangle
        l = a.boundary
        coords = [c for c in l.coords]
        segments = [shapely.geometry.LineString([a, b]) for a, b in zip(coords,coords[1:])]
        longest_segment = max(segments, key=lambda x: x.length)

        p1, p2 = [c for c in longest_segment.coords]
        angle = math.degrees(math.atan2(p2[1]-p1[1], p2[0]-p1[0]))
        return angle
    
    def __get_area(self):
        '''
        calculate area for our polygons and append to the geo_df
        '''
        # If it we have the df
        # Get the area and append to the df
        if self.geo_df is not None :
            # Calculate the area
            self.geo_df['area(square meter)'] = ((self.geo_df['geometry'].area)) # In sequare meter #/10.764
        else:
            print("No dataframe acquired, yet. Use get_geo_df")

        return self.geo_df
    
    
    def __get_azimuth(self):
        '''Calculate the azimuth angle'''
        # Make sure we already have the geo_df calculated
        list_azimuth = []
        if self.geo_df is not None :
            for i in range(len(self.geo_df)):
                g = self.geo_df.iloc[i].geometry
                angle = self.__calc_azimuth(g)
                list_azimuth.append(angle)
                
            self.geo_df['Roof_Azimuth'] = list_azimuth
        
        else: 
             print("No dataframe acquired, yet. Use get_geo_df")
        
        return self.geo_df
    
    def __get_px_coords(self):
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
    
    def __get_geo_coords(self):
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
                rows = points[0]
                cols = points[1]
        
                (px, py) = rio.transform.xy(self.tif_img.transform, rows, cols, offset='center')
                poly = (px, py)
                geo_poly.append(poly)
        
            geo_coords.append(geo_poly)
    
        geo_polys = []    
        for poly in geo_coords:
            geo_polys.append(Polygon(poly))
        
        # Add the new polygons to our gdf
        self.geo_df['polygons'] = geo_polys
        
        # Create a dataframe ans set the geometry to the newly created polygons
        
        # store the pixel coords
        pix_polys =  self.geo_df.geometry
        
        gdf = gpd.GeoDataFrame(self.geo_df, geometry = self.geo_df.polygons)  
        gdf['pix_polygons'] = pix_polys
        
        # Set the original crs
        # Reproject to ge the real coords on the map 
        gdf = gdf.set_crs(str(self.tif_img.crs))
        
        self.geo_df = gdf
        # Apply the new projection
        gdf = gdf.to_crs(epsg=2955) #5703
        self.geo_df = gdf
        return self.geo_df
    
    def __apply_projection(self):
        '''Apply the new projection to a geodataframe'''
        self.geo_df =  self.geo_df.to_crs(epsg=4326) #5703 in meter
        return self.geo_df
    
    def __get_lat_long(self):
        """Function to get coordinates in latitude and longitude degress as a column in dataframe"""
        # Find the center of the polygons
        self.geo_df['center_point'] = self.geo_df['geometry'].centroid
        #Extract lat and lon from the centerpoint (This is extra)
        self.geo_df["latitude"] = self.geo_df.center_point.map(lambda p: p.x)
        self.geo_df["longitude"] = self.geo_df.center_point.map(lambda p: p.y)
        self.geo_df['points'] = [Point(xy) for xy in zip( self.geo_df['longitude'],  self.geo_df['latitude'])]
        self.geo_df = self.geo_df.drop(['center_point', 'polygons', 'pix_polygons', 'pixel_Center_point'], axis = 1)
        
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
        
        for i in range(len(obj)):
            # Check the prediction score
            if obj[i]['score'] > self.threshold:
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
                            'pixel_x': self.geo_df.iloc[i].pixel_x,
                            'pixel_y': self.geo_df.iloc[i].pixel_y,
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
        for i in range(len(obj)):
            if obj[i]['score'] > self.threshold: 
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


