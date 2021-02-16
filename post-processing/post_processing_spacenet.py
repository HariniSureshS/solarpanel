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
import csv
import pandas as pd
from skimage import measure

#!/usr/bin/env python
# coding: utf-8

# In[7]:

#!/usr/bin/env python
# coding: utf-8

class Post_Process:
    def __init__(self, addr_loc, coco_file, crs_value='EPSG:4326'):
        '''
        class takes model prediction (coco json format)
        and tile_prediction(full path) of the model and location , for post processing
        Returns, binary mask and dataframe with area and oriantation
        '''
        self.coco_file = coco_file
        self.add_loc = addr_loc
        self.geo_df = None
        self.building_df = None
        self.bin_mask = None
        self.result_dict = []
        self.crs_value = crs_value
        # 
        #self.tif_img = rio.open(tile_prediction)
    
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
            obj = json.loads(obj)
        
        #poly_loc_list = []
        ## iterate and decode segmentations
        self.bin_mask = np.zeros(np.array(obj['pred_masks'][0]).shape)

        for i in range(len(obj['boxes'])):
            seg_dict = {}
            seg_dict = {'scores': obj['scores'][i], 'pred_masks' : np.array(obj['pred_masks'][i], dtype=int)}
        
            #poly = mask_util.decode(seg_dict)[:, :]
            self.bin_mask += seg_dict['pred_masks']
            
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
        geo_df_meter = self.geo_df.copy()
        geo_df_meter = self.geo_df.to_crs('EPSG:3763') #5703
        # If it we have the df
        # Get the area and append to the df
        if geo_df_meter is not None :
            # Calculate the area
            geo_df_meter['area(square meter)'] = ((geo_df_meter['geometry'].area)) # In sequare meter #/10.764
            # Back to original 
            self.geo_df['area(square meter)'] =  geo_df_meter['area(square meter)']
        else:
            print("No dataframe acquired, yet. Use get_geo_df")
        
        return self.geo_df
    
    
    def get_azimuth(self):
        '''Calculate the azimuth angle'''
        # Make sure we already have the geo_df calculated
        list_azimuth = []
        if self.geo_df is not None :
            geo_df_meter = self.geo_df.copy()
            geo_df_meter = geo_df_meter.to_crs('EPSG:3763')

            for i in range(len(geo_df_meter)):
                g = geo_df_meter.iloc[i].geometry
                angle = self.calc_azimuth(g)
                list_azimuth.append(angle)
                
            geo_df_meter['Roof_Azimuth'] = list_azimuth
        
        else: 
             print("No dataframe acquired, yet. Use get_geo_df")
        self.geo_df['Roof_Azimuth'] =  geo_df_meter['Roof_Azimuth']
        return self.geo_df
    
    def get_px_coords_df(self):
        '''Convert ourbinary mask to polygons'''       
        if self.bin_mask is not None:
            contours = measure.find_contours(self.bin_mask, 0.5)
            len(contours)
            px_polys = []
            for contour in contours:
                if len(contour) > 3: # Check for valid polygons points
                    #poly.is_valid
                    poly = Polygon(contour).simplify(1.0)
                    if Polygon(contour).simplify(1.0).is_valid:
                        px_polys.append(poly)

            data = []
            for i in range(len(px_polys)):
                data_dict = {}
                data_dict = {'value': i, 'geometry' :px_polys[i]}
                data.append(data_dict)

            # Create dataFrame
            self.geo_df = gpd.GeoDataFrame(data)
            
        else:
            print("Claclulate binary mask first using get_bin_mask")
        
        self.geo_df = self.geo_df.drop(['value'], axis = 1) 
        return self.geo_df
    
    '''def get_px_coords_df(self):
        if self.bin_mask is not None:
            
            self.geo_df = sol.vector.mask.mask_to_poly_geojson(pred_arr=self.bin_mask, 
                                                 reference_im=self.tif_img,  
                                                 min_area=1, simplify=True) #min_area=1
            
            self.geo_df = self.geo_df.drop(['value'], axis = 1) 
            
            
        else:
            print("Claclulate binary mask first using get_bin_mask")
            
        return self.geo_df'''
    
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
        
                (px, py) = rio.transform.xy(self.crs_value, x, y, offset='center')
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
    
    
    def zoom_in(self, tile_pred, num_px):
        '''Takes georefrenced lat/long coors, number of pixel as integer, and tiff image(dir)
        Returns a zoomed in image to that specific location and return the image as a numpy array
        Also clip the geo dataframe accordingly'''
        point = Point(self.add_loc[1:])
        #

        # Convert to GeoSeries
        gdf = gpd.GeoSeries(point)
        # Set projection (This will help get the correct pixel values)
        gdf.set_crs('EPSG:4326')

        # Get the pixel coords 
        row, col = rio.transform.rowcol(self.crs_value, self.add_loc[2], self.add_loc[1])
        # Set the image edges
        p1, p2, p3, p4 = row-num_px, row+num_px, col-num_px, col+num_px
        # Read the model prediction (the whole tile) png
        image = cv2.imread(tile_pred)
        # Select the edges
        im = image[p1:p2, p3:p4, :]
        
        # Select the building from the dataframe
        geo_df_full_c = self.geo_df.copy()
        # Reproject to get the correct distance
        geo_df_full_c = geo_df_full_c.to_crs(5703)
        polygon_index = geo_df_full_c.distance(point).sort_values().index[0]
        self.building_df = pd.DataFrame(geo_df_full_c.iloc[polygon_index])
        ########################################
            
        
        return im, self.building_df

    
    
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
            obj = json.loads(obj)
        for i in range(len(self.geo_df)):
            seg_dict = {}
            seg_dict = {'boxes': obj['boxes'][i], 'scores' : obj['scores'][i], 'pred_masks':obj['pred_masks'][i]}
            seg_dict['image_name'] = self.img_name 
            seg_dict['geometry'] = (np.asarray(self.geo_df.iloc[i].geometry.exterior.coords)).tolist()
            seg_dict['longitude'] =  self.geo_df.iloc[i].longitude
            seg_dict['latitude'] = self.geo_df.iloc[i].latitude
            seg_dict['area(square meter)'] = self.geo_df.iloc[i]['area(square meter)']
            seg_dict['Roof_Azimuth'] = self.geo_df.iloc[i].Roof_Azimuth       
                                    
            self.result_dict.append(seg_dict)

            with open(filename, 'w') as outfile:
                json.dump(self.result_dict, outfile)
                    
        
    # Helper method to save our list of dictionaries as a csv file
    def save_to_csv(self, file_name):
        '''Takes a file name with csv extension'''
        keys = self.result_dict[0].keys()
        with open(file_name, 'w', newline='')  as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows( self.result_dict)