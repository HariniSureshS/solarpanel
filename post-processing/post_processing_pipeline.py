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
from PIL import Image
from affine import Affine
from matplotlib.path import Path
from shapely import affinity
import imutils






#!/usr/bin/env python
# coding: utf-8

# In[7]:

#!/usr/bin/env python
# coding: utf-8

class Post_Process:
    def __init__(self, building_tile, xycoords, coco_file, crs_value='EPSG:4326'):
        '''
        class takes model prediction (coco json format)
        and tile_prediction(full path) of the model and location , for post processing
        Returns, binary mask and dataframe with area and oriantation
        '''
        self.coco_file = coco_file
        self.add_loc = xycoords
        self.geo_df = None
        self.building_df = None
        self.bin_mask = None
        self.result_dict = []
        self.crs_value = crs_value
        self.tile_prediction = building_tile
        
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
            seg_dict = {'scores': obj['scores'][i], 'pred_masks' : np.array(obj['pred_masks'][i]).astype('int')}
        
            #poly = mask_util.decode(seg_dict)[:, :]
            self.bin_mask += seg_dict['pred_masks']
            
        return  self.bin_mask
    
    
    def process_bin_mask(self, mask, bbox):
        '''Helper function that takes a binary mask (PIL Image)
        and a bounding box. Returns the croped mask with paddings'''
        # Crop out our building mask from the tile
        mask = Image.fromarray(mask)
        
        mask_crop=mask.crop(bbox)
        mask_crop = np.asarray(mask_crop)
        
        # add padding of 3 black pixels so we end up with valid polygons
        mask_crop_pad = np.pad(mask_crop , (3, 3), 'constant')
        #Thresholding our mask
        mask_ = mask_crop_pad.copy()
       
        return mask_.astype('int')
    
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
    
    def get_area(self, df):
        '''
        calculate area for our polygons and append to the geo_df
        '''
        # reproject to meter coordinates
        geo_df_meter = df.copy()
        geo_df_meter = geo_df_meter.to_crs('EPSG:3763') #5703
        # If it we have the df
        # Get the area and append to the df
        if geo_df_meter is not None :
            # Calculate the area
            area = ((geo_df_meter['geometry'].area) * 10.7639) # In sequare meter /10.764

        else:
            print("No dataframe acquired, yet. Use get_geo_df")
        
        return area
    
    def get_px_coords_df(self):
        '''Convert ourbinary mask to polygons'''
        if self.bin_mask is not None:
            mask  = self.bin_mask.copy()
            mask = mask.astype('int')
            contours = measure.find_contours(self.bin_mask, 0.5)
            
            #len(contours)
            px_polys = []
            for contour in contours:
                if len(contour) > 3: # Check for valid polygons points
                    #poly.is_valid
                    poly = Polygon(contour).simplify(1.0)
                    if Polygon(contour).simplify(1.0).is_valid:
                        px_polys.append(poly)

            data = []
            boxes = self.get_bboxes()
           
            for i in range(len(px_polys)):
                data_dict = {}
                data_dict = {'value': i, 'geometry' :px_polys[i], 'bbox' : boxes[i]}
                data.append(data_dict)
            
            # Create dataFrame
            self.geo_df = gpd.GeoDataFrame(data)
            
        else:
            print("Claclulate binary mask first using get_bin_mask")
        
        self.geo_df = self.geo_df.drop(['value'], axis = 1) 
        return self.geo_df, contours
   
    def get_px_coords(self):
        """Function to get geo coordinates given a pixel coordinates"""
        xs_list = []
        ys_list = []
        # Get thelist of pixel lat/long
        # First we get the pixel lat/long values
        self.geo_df['px_coords'] = self.geo_df['geometry'].centroid
        
        ''' 
        #Extract lat and lon from the centerpoint
        self.geo_df["pixel_x"] = self.geo_df.pixel_Center_point.map(lambda p: p.x)
        self.geo_df["pixel_y"] = self.geo_df.pixel_Center_point.map(lambda p: p.y)'''
              
        return self.geo_df


    def get_bboxes(self):
        '''Get the bbox list from our prediction json'''
        # Read the coco file and get the boxes
        ## Read and parse our json file
        with open(self.coco_file, 'r') as my_file:
            data = my_file.readlines()
            # Parse file
            obj = json.loads(data[0])
            obj = json.loads(obj)
        bbox_list = []
        for box in obj['boxes']:
            bbox_list.append(box)
        return bbox_list
    
    
    def extract_building(self, px_df, target_building= 'D:/target_building.png'):
        ''' takes pixel coordiantes dataframe and 
        Get the building closest to our building (lat/long/center) '''
        # Find the center point of an image
        im=cv2.imread(self.tile_prediction)
        center_point = [im.shape[0]/2, im.shape[1]/2]
        
        # Check the bounding box containing our center point
        target_bbox = [0, 0, 0, 0]
        for bbox in np.array(px_df.bbox):
            if (bbox[0] < center_point[0] < bbox[2]) and (bbox[1] < center_point[1] < bbox[3]):
                target_bbox = bbox
                
        return target_bbox
    
    def get_bin_mask_poly(self, mask):
        #Convert ourbinary mask to polygons      
        if mask is not None:
            contours = measure.find_contours(mask, 0.5)
            px_polys = []
            for contour in contours:
                if len(contour) > 3: # Check for valid polygons points
                    #poly.is_valid
                    poly = Polygon(contour).simplify(1.0)
                    if poly.is_valid:
                        px_polys.append(poly)

            data = []
           
            for i in range(len(px_polys)):
                data_dict = {}
                data_dict = {'value': i, 'geometry' :px_polys[i]}
                data.append(data_dict)
            
            # Create dataFrame
            px_df = gpd.GeoDataFrame(data)
            
        else:
            print("Claclulate binary mask first using get_bin_mask")
        
        #px_df = px_df.drop(['value'], axis = 1) 
        return px_df, contours
   
        
    
    def get_azimuth(self, df):
        #Calculate the azimuth angle
        # Make sure we already have the geo_df calculated
        list_azimuth = []
        if df is not None :
            geo_df_meter = df.copy()
            geo_df_meter = geo_df_meter.to_crs('EPSG:5703')

            for i in range(len(geo_df_meter)):
                g = geo_df_meter.iloc[i].geometry
                angle = self.calc_azimuth(g)
                list_azimuth.append(angle)       
        
        else: 
             print("No dataframe acquired, yet. Use get_geo_df")
        
        return list_azimuth
    
    
    
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
            seg_dict = {'boxes': obj['boxes'][i], 'pred_masks':obj['pred_masks'][i]}
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
            
    def get_geo_coords(self, df, tif_img):
        #Convert the each pixel point 
        #to a georefrenced point with lat/long coordinates
        # Now convert the pixel row/col to lat/long
        # unravel GDAL affine transform parameters
        coords = [list(poly.exterior.coords) for poly in df.geometry]
        geo_poly = []
        geo_coords = []

        for poly in coords:
            geo_poly = []
            #print(poly)
            for points in poly:
                x = points[0]
                y = points[1]
        
                (px, py) = rio.transform.xy(tif_img.transform, x, y, offset='center')
                poly = (px, py)
                geo_poly.append(poly)
        
            geo_coords.append(geo_poly)
    
        geo_polys = []    
        for poly in geo_coords:
            geo_polys.append(Polygon(poly))
        
        # Add the new polygons to our gdf
        #self.geo_df['px_polygonss'] = self.geo_df['geometry'] # The old pixel polygons
        gdf = df.copy()
        gdf['geometry'] = geo_polys # assign the new polygons
        gdf = gdf.set_crs(self.crs_value) #
        return gdf
    
    
    def normalize_orientation(self, building_img, azimuth):
        '''Takes an image of a building and azimuth angle
        Returns the image aligned with the x or y axis'''
        rotated = building_img
        if azimuth == 0 or np.abs(azimuth) == 90:
            rotated = building_img
        if azimuth > 0 :
            #x = azimuth - 90
            rotated = imutils.rotate_bound(building_img, -azimuth)
            
        elif azimuth < 0:
            x = 90 + -1 *azimuth
            rotated = imutils.rotate_bound(building_img, +azimuth)
            
        return rotated
    
    #Create bin mask from a polygon
    def poly_to_mask(self, polygon, mask):
        polygon_ = polygon #polygon.buffer(10, join_style=2).buffer(-10.0, join_style=2)
        p = np.array(polygon_.exterior.coords.xy)

        p = np.swapaxes(p, 0, 1)
        nx, ny = mask.shape[0] , mask.shape[1]
        poly_verts = p

        # Create vertex coordinates for each grid cell...
        # (<0,0> is at the top left of the grid in this system)
        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
        x, y = x.flatten(), y.flatten()

        points = np.vstack((x,y)).T

        path = Path(poly_verts)
        grid = path.contains_points(points)
        grid = grid.reshape((ny,nx))
        grid = grid.astype('uint8')
    
        return grid 

      