#!/usr/bin/env python
# coding: utf-8

import json
import os
import pathlib
import time
import pandas as pd
import geopandas as gpd
import osgeo #needed if running notebook on Windows
import requests
from requests.auth import HTTPBasicAuth

pre_geom_path = os.path.join(r'data','byCell_shp')
geom_path = os.path.join(r'data','byCell')
out_dir = os.path.join(r'data','Img_Lists')

# API Key stored as an env variable
PLANET_API_KEY = os.getenv("PL_API_KEY")
if PLANET_API_KEY is None:
    PLANET_API_KEY = '12345'
orders_url = 'https://api.planet.com/compute/ops/orders/v2'
data_url = "https://api.planet.com/data/v1"

def convert_shp_to_geojson(filename, pre_geom_path, geom_path):
    shp_in = gpd.read_file(os.path.join(pre_geom_path,filename), encoding="utf-8")
    #Reproject to EPSG 4326 if not already
    try:
        data_proj = shp_in.copy()
        data_proj["geometry"] = data_proj["geometry"].to_crs(epsg=4326)
        data_proj.to_file(os.path.join(geom_path, str(filename).replace(".shp", ".geojson")),driver="GeoJSON")
    except Exception as e:
        print(e)
        
def convert_geojson_to_geom(geojson,polyid):
    try:
        with open(geojson) as jsonfile:
            data = json.load(jsonfile)
            aoi_geom = data["features"][polyid]["geometry"]
            #print(aoi_geom)
    except Exception as e:
        print("Please check GeoJSON: Could not parse coordinates")
        print(e)
    return aoi_geom

def make_planet_filters(aoi_geom, start_date, end_date, cloud_max, item_type):
    '''
    dates should be YYYY-MM-DD
    '''
    geometry_filter = {
      "type": "GeometryFilter",
      "field_name": "geometry",
      "config": aoi_geom
    }

    
    date_range_filter = {
      "type": "DateRangeFilter",
      "field_name": "acquired",
      "config": {
        "gte":"{}-{}-{}T00:00:00Z".format(start_date.split("-")[0],start_date.split("-")[1],start_date.split("-")[2]),
        "lte":"{}-{}-{}T00:00:00Z".format(end_date.split("-")[0],end_date.split("-")[1],end_date.split("-")[2])
      }
    }

    cloud_cover_filter = {
      "type": "RangeFilter",
      "field_name": "cloud_cover",
      "config": {
        "gte": 0.0
      }
    }

    combined_filter = {
      "type": "AndFilter",
      "config": [geometry_filter, date_range_filter, cloud_cover_filter]
    }

    item_type = "PSScene"

    # API request object
    search_request = {
      "item_types": [item_type], 
      "filter": combined_filter
    }
    
    return search_request

def planet_search_results(search_request, out_dir, out_name) :
    output_dict = {}
    search_result = requests.post('https://api.planet.com/data/v1/quick-search', auth=HTTPBasicAuth(PLANET_API_KEY, ''), json=search_request)
    #print(search_result.json())
    #print(search_result.json()['features'])
    for feature in search_result.json()['features']:
        output_dict[feature['id']]={}
        output_dict[feature['id']]['acquired'] = feature['properties']['acquired']
        if 'cloud_percent' in feature['properties']:
            output_dict[feature['id']]['clouds'] = feature['properties']['cloud_percent']
            output_dict[feature['id']]['clear'] = feature['properties']['clear_percent']
            output_dict[feature['id']]['heavy_haze'] = feature['properties']['heavy_haze_percent']
            output_dict[feature['id']]['light_haze'] = feature['properties']['light_haze_percent']
        output_dict[feature['id']]['sensor'] = feature['properties']['instrument']
        output_dict[feature['id']]['quality'] = feature['properties']['quality_category']
    out_df = pd.DataFrame.from_dict(output_dict,orient='index')
    out_df.to_csv(os.path.join(out_dir, out_name),index=False)
    return out_df

def list_overlapping_imgs_in_dir(img_dir, aoi_file, aoi_id): 
    print('finding overlapping images...')
    if aoi_file.endswith('.csv'):
        locations = pd.read_csv(aoi_file)
        loc_id = locations['unique_id'][aoi_id]
        poly_minlon = locations['minLon'].values.tolist()[aoi_id]
        poly_minlat = locations['minLat'].values.tolist()[aoi_id]
        poly_maxlon = locations['maxLon'].values.tolist()[aoi_id]
        poly_maxlat = locations['maxLat'].values.tolist()[aoi_id]
        bb = [poly_minlon, poly_minlat, poly_maxlon, poly_maxlat]
    elif aoi_file.endswith('.gpkg'):
        aois = gpd.read_file(aoi_file)
        bb = aois.query(f'UNQ == {aoi_id}').geometry.total_bounds
    elif aoi_file.endswith('.shp'):
        aois = gpd.read_file(aoi_file)
        bb = aois.geometry.bounds
        poly_minlon = bb.minx
        poly_minlat = bb.miny
        poly_maxlon = bb.maxx
        poly_maxlat = bb.maxy
    poly_bbox = box(poly_minlon, poly_minlat, poly_maxlon,poly_maxlat)

    overlapping_images = []
    # get all unique base_ids:
    img_ids = [os.path.basename(f)[:21] for f in  os.listdir(img_dir)]
    print(img_ids)
    img_ids_unq = list(set(img_ids))
    print(img_ids_unq)
    
    for idx in img_ids_unq:
        image_set = [f for f in os.listdir(img_dir) if os.path.basename(f)[:21] == idx]
        id_checked = False
        for fp in image_set:
            if ".xml" in fp:
                #Read metadata to check for overlap in footprint:
                try:
                    metaparse = minidom.parse(os.path.join(img_dir,fp))
                    img_minlon = float(metaparse.getElementsByTagName("ps:bottomLeft")[0].getElementsByTagName("ps:longitude")[0].firstChild.data)
                    img_minlat = float(metaparse.getElementsByTagName("ps:bottomLeft")[0].getElementsByTagName("ps:latitude")[0].firstChild.data)
                    img_maxlon = float(metaparse.getElementsByTagName("ps:topRight")[0].getElementsByTagName("ps:longitude")[0].firstChild.data)
                    img_maxlat = float(metaparse.getElementsByTagName("ps:topRight")[0].getElementsByTagName("ps:latitude")[0].firstChild.data)
                    #print ("For image: minlon = {}, minlat = {}, maxlon = {}, maxlat = {}".format(img_minlon, img_minlat, img_maxlon, img_maxlat))
                    #check if raster overlaps polygon (ignore if it doesn't):
                    id_checked = True
                except:
                    id_checked = False
                    pass
            if id_checked == False:
                for fp in image_set:
                    if fp.endswith('.tif') and 'AnalyticMS' in fp and 'utm' not in fp:
                        with rio.open(os.path.join(img_dir,fp)) as src:
                            bds = src.bounds
                            img_minlon = bds[0]
                            img_minlat = bds[1]
                            img_maxlon = bds[2]
                            img_maxlat = bds[3]
                            id_checkes = True
            if id_checked == True:
                if (img_maxlat < poly_minlat or img_maxlon < poly_minlon or img_minlon > poly_maxlon or img_minlat > poly_maxlat):
                    # verified empty intersection
                    pass
                else:
                    print ("found an intersection with image {}".format(idx))
                    overlapping_images.append(idx)
            else:
                print(f'problem finding bounds for {idx}')
    
    return overlapping_images
    
def get_obs_stats_for_poly(img_db, grid_dict, poly_id):
    img_db.reset_index(inplace=True)
    img_db.rename(columns={'index': 'id'},inplace=True)
    img_db['acquired'] = pd.to_datetime(img_db['acquired'])
    img_db.sort_values(by='acquired',axis=0,inplace=True)
    img_db['next_obs'] = img_db['acquired'].shift(-1)
    img_db['obs_interval'] = (img_db['next_obs'] - img_db['acquired']) / pd.Timedelta(seconds=1)

    grid_dict[poly_id] = {}
    grid_dict[poly_id]['UNQ'] = poly_id.split('_')[1]
    grid_dict[poly_id]['max_gap_hrs'] = img_db['obs_interval'].max() /3600
    grid_dict[poly_id]['avg_gap_hrs'] = img_db['obs_interval'].mean() /3600
    grid_dict[poly_id]['med_gap_hrs'] = img_db['obs_interval'].median() /3600
    grid_dict[poly_id]['gaps_gt2days'] = img_db['obs_interval'].gt(172800).sum()
    grid_dict[poly_id]['gaps_gt3days'] = img_db['obs_interval'].gt(259200).sum()
    grid_dict[poly_id]['gaps_gt4days'] = img_db['obs_interval'].gt(345600).sum()
    grid_dict[poly_id]['gaps_gt5days'] = img_db['obs_interval'].gt(432000).sum()
    #print('there are {} gaps >2 days, {} gaps >3 days, {} gaps >4 days, and {}gaps > 5days. the biggest gap is {} days.'\
    #      .format(gaps_gt2days, gaps_gt3days, gaps_gt4days, gaps_gt5days, max_gap_hrs/24))
    print(grid_dict[poly_id])
    
    return grid_dict

def get_obs_stats_for_multipoly(start_date, end_date, image_source, cloud_max, sensor, item_type, out_dir, geom_path, pre_geom_path=None):
    '''
    if {image_source} == Web, checks stats from Planet online directly
    if {image_source} is a folder, gets stats from images already downloaded to a folder
    
    if querying from web directly, can convert shapefiles (in pre_geom_path) to geojsons (in geom_path) and then to Planet geom object. 
    If geometry objects already exist as geojsons or want to use shapefiles (not querying Planet), use pre_geom_path=None
    '''
        
    grid_dict = {}
    
    if sensor == 'Planet':
        if pre_geom_path:
            if os.path.isfile(pre_geom_path):
                df = gpd.read_file(pre_geom_path)
                print('there are {} AOIs in AOI file'.format(df.shape[0]))
                for p in range(1,df.shape[0]+1):
                    print(df.query(f'UNQ == {p}').geometry)
                    df.query(f'UNQ == {p}').geometry.to_file(os.path.join(geom_path,f'UNQ_{p}.geojson'), driver='GeoJSON')
            elif os.path.isdir(pre_geom_path):
                shapefiles = [f for f in os.listdir(pre_geom_path) if f.endswith(".shp")]
                if len(shapefiles) > 0:
                    for sf in shapefiles:
                        convert_shp_to_geojson(sf,pre_geom_path,geom_path)
                    print(f"Exported {len(shapefiles)} shapefiles to geoJSONs, which are in: {geom_path}")
        aois = [g for g in os.listdir(geom_path) if g.endswith(".geojson")]
        print(len(aois))
        for aid,aoi in enumerate(aois):
            aoi_id = aoi.split(".")[0]
            aoi_geom = convert_geojson_to_geom(os.path.join(geom_path,aoi),0)
            if image_source == 'Web':   
                search_request = make_planet_filters(aoi_geom, start_date, end_date, cloud_max, item_type)
                outname = "PlanetResults_{}to{}_for_AOI_{}.csv".format(start_date, end_date, aoi_id)
                result_db = planet_search_results(search_request, out_dir, outname)
                print(f"saved list for {outname}")
                get_obs_stats_for_poly(result_db, grid_dict, aoi_id)
            else:
                #aois = [g for g in os.listdir(geom_path) if g.endswith(".shp") or g.endswith(".geojson")]
                images = [img for img in os.listdir(image_source) if img.endswith(".tif")]
                overlapping = list_overlapping_imgs_in_dir(img_dir, aoi_file, aoi_id)
                ## TODO: finish this!
            
    grid_db = pd.DataFrame.from_dict(grid_dict,orient='index')
    db_name = '{}_obs_stats_{}to{}_ALLCELLS.csv'.format(sensor,start_date,end_date)
    grid_db.sort_values(by='max_gap_hrs',axis=0,inplace=True)
    grid_db.to_csv(os.path.join(out_dir, db_name))
    
    return grid_dict

def get_obs_stats_for_pt(img_dir):
    print('working on this method...')
    # do this?