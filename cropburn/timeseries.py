
import os
from pathlib import Path
import rasterio as rio
from rasterio import plot
from rasterio.plot import show
import matplotlib.pyplot as plt
import shutil
import tempfile
import json
import random
import datetime
import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import Proj, transform
from pyproj import CRS
from shapely.geometry import box
from shapely.geometry import shape
from shapely.geometry import MultiPoint
from shapely.geometry import Point
from shapely.geometry import Polygon
import pyproj
from pyproj import Proj, transform
from statistics import mean
import xarray as xr
from ipyleaflet  import Map, GeoData

def get_img_date(img, image_type, data_source=None):
   
    if '.' in img:
        img_base = os.path.basename(img)
    else:
        img_base = img
        
    if image_type == 'Smooth':  #Expects images to be named YYYYDDD
        YYYY = int(img_base[:4])
        doy = int(img_base[4:7])
    elif image_type in ['Planet','Sentinel','Landsat','brdf']:
        if image_type == 'Planet':
            YYYYMMDD = img_base[:8]
        elif image_type == 'Sentinel' and 'brdf' not in str(img_base):
            YYYYMMDD = img_base.split('_')[2][:8]
        else:
            YYYYMMDD = img.split('_')[3][:8]
          
        YYYY = int(YYYYMMDD[:4])
        MM = int(YYYYMMDD[4:6])
        DD = int(YYYYMMDD[6:8])
   
        ymd = datetime.datetime(YYYY, MM, DD)
        doy = int(ymd.strftime('%j'))
        
    else:
        print ('Currently valid image types are Smooth,Planet, Sentinel,Landsat. You put {}'.format(image_type))     
        
    return YYYY, doy
    
def convert_and_print_coord_list(coord_list,img_crs, out_dir):
    coord_list_lat = [c[1] for c in coord_list]
    coord_list_lon = [c[0] for c in coord_list]
    ### Convert list of coordinates back to original CRS and print to file:
    coord_listX = []
    coord_listY = []
    transformer = pyproj.Transformer.from_crs("epsg:4326", img_crs)
    for pt in transformer.itransform(coord_list):
        print('{:.3f} {:.3f}'.format(pt[0],pt[1]))
        coord_listX.append(pt[0])
        coord_listY.append(pt[1])
        coords = {'XCoord':coord_listX,'YCoord':coord_listY, 'lat':coord_list_lat,'lon':coord_list_lon}
    coorddb = pd.DataFrame(coords)
    coorddb = coorddb.astype({'XCoord':'float','YCoord':'float', 'lat':'float', 'lon':'float'})
    coord_path = os.path.join(out_dir,'SelectedCoords.csv')
    coorddb.to_csv(coord_path, sep=',', na_rep='NaN', index=True)
    return coord_path

def get_values_at_coords(coord_list, coord_crs, img, bands):
    
    ptsval = {}
    if isinstance(coord_list, pd.DataFrame):
        ptsdf = coord_list
    else:
        ptsdf = pd.read_csv(coord_list)

    pts = gpd.GeoDataFrame(ptsdf,geometry=gpd.points_from_xy(ptsdf.XCoord,ptsdf.YCoord),crs=coord_crs)
    xy = [pts['geometry'].x, pts['geometry'].y]
    coords = list(map(list, zip(*xy)))
    
    if img.endswith('.tif'):
        with rio.open(img, 'r') as src:
            for b in bands:
                ptsval[b] = [sample[b-1] for sample in src.sample(coords)]
    '''
    elif img.endswith('.nc'): 
        xrimg = xr.open_dataset(img)
        for b in bands:
            xr_val = xrimg[b.where(xrimg[b] < 10000)]

            vals=[]
            for index, row in pts.iterrows():
                thispt_val = xr_val.sel(x=pts['geometry'].x[index],y=pts['geometry'].y[index], method='nearest', tolerance=30)
                this_val = thispt_val.values
                vals.append(this_val)
                ptsval[b] = vals
    '''
    return ptsval

def get_coord_at_rowcol (img, spec_index, row, col):
    df = explore_band(img, spec_index)
    test_samp = df[row,col]
    print('Test Samp at coords x={}, y={} is {}'.format(test_samp['x'].values,test_samp['y'].values, test_samp.values))
    return test_samp['x'].values, test_samp['y'].values

def get_val_at_XY(img, spec_index, xcoord, ycoord):
    df = explore_band(img, spec_index)
    one_point = df[spec_index].sel(x=xcoord,y=ycoord, method='nearest', tolerance=15)
    print('value at {},{}={}'.format(xcoord,ycoord,one_point.values))
    return one_point.values

def get_pts_in_grid (grid_file, grid_cell, pt_file):
    '''
    loads point file (from .csv with 'XCoord' and 'YCoord' columns) and returns points that overlap a gridcell
    as a geopandas GeoDataFrame. Use this if trying to match/append data to existing sample points
    rather than making a new random sample each time (e.g. if matching Planet and Sentinel points)
    Note that crs of 'XCoord' and 'YCoord' in {pt_file} needs to match crs of {grid_file}
    If {pt_file} is 'centroids', will get center point of grid cell
    '''
    out_path = Path(grid_file).parent

    with tempfile.TemporaryDirectory(dir=out_path) as temp_dir:
        temp_file = Path(temp_dir) / Path(grid_file).name
        shutil.copy(grid_file, temp_file)
        df = gpd.read_file(temp_file)
        crs_grid = df.crs
    print('grid is in: ', crs_grid)
    
    cl = df.query(f'UNQ == {grid_cell}')
    bb = cl.geometry.total_bounds

    grid_bbox = box(bb[0],bb[1],bb[2],bb[3])
    grid_bounds = gpd.GeoDataFrame(gpd.GeoSeries(grid_bbox), columns=['geometry'], crs=crs_grid)
    print(grid_bounds)
    
    if pt_file == "centroids":
        pts_in_grid = gpd.GeoDataFrame(geometry=gpd.GeoSeries(cl.centroid))
        print(f'centroid point: {pts_in_grid}')
    else:
        ptsdf = pd.read_csv(pt_file, index_col=0)
        pts = gpd.GeoDataFrame(ptsdf,geometry=gpd.points_from_xy(ptsdf.XCoord,ptsdf.YCoord),crs=crs_grid)

        pts_in_grid = gpd.sjoin(pts, grid_bounds, op='within')
        pts_in_grid = pts_in_grid.loc[:,['geometry']]

        print("Of the {} ppts, {} are in gridCell {}". format (pts.shape[0], pts_in_grid.shape[0],grid_cell))
        
    #Write to geojson file
    if pts_in_grid.shape[0] > 0:
        pt_clip = Path(os.path.join(out_path,'ptsGrid_'+str(grid_cell)+'.json'))
        pts_in_grid.to_file(pt_clip, driver="GeoJSON")

        return pts_in_grid
        print(pts_in_grid.head(n=5))

def get_polygons_in_grid (grid_file, grid_cell, poly_path):
    '''
    Filters polygon layer to contain only those overlapping selected grid cell (allows for iteration through grid)
    Outputs new polygon set to a .json file stored in the gridcell directory
    '''
    polys = gpd.read_file(poly_path)
    out_path = Path(grid_file).parent

    with tempfile.TemporaryDirectory(dir=out_path) as temp_dir:
        temp_file = Path(temp_dir) / Path(grid_file).name
        shutil.copy(grid_file, temp_file)
        df = gpd.read_file(temp_file)
        crs_grid = df.crs

    bb = df.query(f'UNQ == {grid_cell}').geometry.total_bounds

    grid_bbox = box(bb[0],bb[1],bb[2],bb[3])
    grid_bounds = gpd.GeoDataFrame(gpd.GeoSeries(grid_bbox), columns=['geometry'], crs=crs_grid)
    polys_in_grid = gpd.overlay(grid_bounds, polys, how='intersection')

    print("Of the {} polygons, {} are in gridCell {}". format (polys.shape[0], polys_in_grid.shape[0],grid_cell))

    #Write to geojson file
    if polys_in_grid.shape[0] > 0:
        poly_clip = Path(os.path.join(out_path,'polysGrid_'+str(grid_cell)+'.json'))
        polys_in_grid.to_file(poly_clip, driver="GeoJSON")

        return poly_clip

def calculate_raw_index(band_vals, spec_index):
    '''
    band_vals[0] is 'nir'. 
    '''
    scale = 10000
    if spec_index == 'evi2':
        index_val = scale * 2.5 * ((band_vals[0]/scale - band_vals[1]/scale) / (band_vals[0]/scale + 1.0 + 2.4 * band_vals[1]/scale))
    elif spec_index == 'ndvi':
        index_val = scale * (band_vals[0] - band_vals[1]) / ((band_vals[0] + band_vals[1]) + 1e-9)
    elif spec_index == 'savi':
        lfactor = .5 #(0-1, 0=very green, 1=very arid. .5 most common. Some use negative vals for arid env)
        index_val = scale * (1 + lfactor) * ((band_vals[0] - band_vals[1]) / (band_vals[0] + band_vals[1] + lfactor))
    elif spec_index == 'msavi':
        index_val =  scale/2 * (2 * band_vals[0]/scale + 1) - ((2 * band_vals[0]/scale + 1)**2 - 8*(band_vals[0]/scale-band_vals[1]/scale))**1/2
    elif spec_index == 'ndmi':
        index_val = scale * (band_vals[0] - band_vals[1]) / ((band_vals[0] + band_vals[1]) + 1e-9)
    elif spec_index == 'ndwi':
        index_val = scale * (band_vals[1] - band_vals[0]) / ((band_vals[1] + band_vals[0]) + 1e-9)
    elif spec_index == 'CI':  #lower means more charred
        vissum = (band_vals[1] + band_vals[2] + band_vals[3])
        bg = abs(band_vals[3] - band_vals[2])
        br = abs(band_vals[3] - band_vals[1])
        rg = abs(band_vals[1] - band_vals[2])
        maxdiff1 = np.maximum(bg, br)
        maxdiff = np.maximum(maxdiff1, rg)
        index_val = scale * ((vissum/scale + (maxdiff*15)/scale))
        print(f'index val = {index_val}')
    elif spec_index == 'nir':
        index_val = band_vals[0]
    elif spec_index in ['swir1','swir2','red','green']:
        index_val = band_vals[1]
        
    return index_val

def get_index_vals_at_pts(out_dir, ts_stack, image_type, polys, spec_index, numPts, apply_masks=False, seed=88, load_samp=False, ptgdb=None):
    '''
    Gets values for all sampled points {'numpts'} in all polygons {'polys'} for all images in {'TStack'}
    OR gets values for points in a previously generated dataframe {ptgdb} using loadSamp=True.
    If imageType == 'TS', indices are assumed to already be calculated and
    {'TStack'} is a list of image paths, with basenames = YYYYDDD of image acquisition (DDD is Julien day, 1-365)
    If imageType == 'L1C' images are still in raw .nc form (6 bands) and indices are calculated here
    {'TStack'} is a list of image paths from which YYYYDDD info can be extracted
    output is a dataframe with a pt (named polygonID_pt#)
    on each row and an image index value(named YYYYDDD) in each column
    '''
    if load_samp == False:
        if polys:
            ptsgdb = get_ran_pts_in_polys (polys, numPts, seed)
        else:
            print('There are no polygons or points to process in this cell')
            return None
    elif load_samp == True:
        ptsgdb = ptgdb
        
    if image_type in ['Smooth']:  # these images have only one band
        for img in ts_stack:
            img_date = get_img_date(img, image_type)
            img_name = str(img_date[0])+(f"{img_date[1]:03d}")
        
            with rio.open(img, 'r') as src:
                ptsgdb[img_name] = [sample[0] for sample in src.sample(coords, masked=True)]
        pts_db = ptsgdb.drop(columns=['geometry'])
        
    elif image_type not in ['Planet','Sentinel','Landsat','brdf']: \
            print ('Currently valid image types are Smooth, Planet, Sentinel, Landsat. You put {}'.format(image_type))

    else:
        pts = {}
        #xy = [ptsgdb['geometry'].x, ptsgdb['geometry'].y]
        #coords = list(map(list, zip(*xy)))
        for img in ts_stack:
            img_date = get_img_date(img, image_type)
            img_name = str(img_date[0])+(f"{img_date[1]:03d}")
            if img_name not in pts:
                pts[img_name] = {}
            
            if apply_masks == True or img.endswith('nc'):
                if apply_masks == True:
                    masked = mask_clouds(img, 'udm2csorig')
                    bandnames = ["blue","green","red","nir"]
                    xrimg = masked.to_dataset('band')
                    xrimg = xrimg.rename({i + 1: name for i, name in enumerate(bandnames)})
                else:
                    xrimg = xr.open_dataset(img)
                   
                xr_nir = xrimg['nir'].where(xrimg['nir'] < 10000)                
                if spec_index in ['evi2','msavi','ndvi','savi','CI','red']:
                    xr_red = xrimg['red'].where(xrimg['red'] < 10000)
                if spec_index in ['ndmi','swir1']:
                    xr_swir1 = xrimg['swir1'].where(xrimg['swir1'] < 10000)
                if spec_index in ['ndwi','CI','green']:
                    xr_green = xrimg['green'].where(xrimg['green'] < 10000)
                if spec_index in ['swir2']:
                    xr_swir2 = xrimg['swir2'].where(xrimg['swir2'] < 10000)
                if spec_index in ['CI']:
                    xr_blue = xrimg['blue'].where(xrimg['blue'] < 10000)
                if spec_index in ['nir']:
                    pass
                #else: print('{} is not specified or does not have current method'.format(spec_index))
            
                for ptid, row in ptsgdb.iterrows():
                    coord = row['geometry']
                    print(f'ptid={ptid},coord={coord}')
                    try:
                        thispt_nir = xr_nir.sel(x=coord.x,y=coord.y,method='nearest', tolerance=30)
                        nir_val = thispt_nir.values
                    except KeyError:
                        nir_val = 0
                    if nir_val > 0:
                        if spec_index in ['evi2','msavi','ndvi','savi','CI','red']:
                            thispt_b2 = xr_red.sel(x=coord.x,y=coord.y,method='nearest', tolerance=30)
                            b2_val = thispt_b2.values
                        if spec_index in ['ndmi','swir1']:
                            thispt_b2 = xr_swir1.sel(x=coord.x,y=coord.y,method='nearest', tolerance=30)
                            b2_val = thispt_b2.values
                        if spec_index in ['ndwi','green']:
                            thispt_b2 = xr_green.sel(x=coord.x,y=coord.y,method='nearest', tolerance=30)
                            b2_val = thispt_b2.values
                        if spec_index in ['swir2']:
                            thispt_b2 = xr_swir2.sel(x=coord.x,y=coord.y,method='nearest', tolerance=30)
                            b2_val = thispt_b2.values
                        if spec_index in ['nir']:
                            b2_val = nir_val
                        if spec_index in ['CI']:
                            thispt_b3 = xr_green.sel(x=coord.x,y=coord.y,method='nearest', tolerance=30)
                            b3_val = thispt_b3.values
                            thispt_b4 = xr_blue.sel(x=coord.x,y=coord.y,method='nearest', tolerance=30)
                            b4_val = thispt_b4.values
                        
                        if ptid not in pts[img_name]:
                            pts[img_name][ptid] = []
                        if spec_index == 'CI':
                            index_val = calculate_raw_index([nir_val, b2_val, b3_val, b4_val], spec_index)
                        else:
                            index_val = calculate_raw_index([nir_val, b2_val], spec_index)
                        pts[img_name][ptid].append(int(index_val))
                        #print(pts)
                        
            elif img.endswith('.tif'):
                with rio.open(img, 'r') as src:
                    for ptid, row in ptsgdb.iterrows():
                        coord = row['geometry']
                        try:
                            nir_val = src.read(4)[src.index(coord.x,coord.y)]
                            #print(f'nir_val = {nir_val}')
                        except IndexError as e:  # this excludes everything that does not overlap
                            nir_val = 0
                        if nir_val > 0:
                            if spec_index in ['evi2','msavi','ndvi','savi','red']:
                                b2_val =  src.read(3)[src.index(coord.x,coord.y)]
                            if spec_index in ['ndwi','green']:
                                b2_val =  src.read(2)[src.index(coord.x,coord.y)]
                            elif spec_index in ['nir']:
                                b2_val = nir_val  
                            index_val = calculate_raw_index([nir_val, b2_val], spec_index)
                            
                            if ptid not in pts[img_name]:
                                pts[img_name][ptid] = []
                            pts[img_name][ptid].append(index_val)
                            #print(pts)
            
        # average values where there are multiple observations for the same day (img_name):
        pts_avg = {idx: {key: mean(idx) for key, idx in j.items()} for idx, j in pts.items()}
        pts_db = pd.DataFrame.from_dict(pts_avg)

    pd.DataFrame.to_csv(pts_db,os.path.join(out_dir,'ptsgdb.csv'), sep=',', index=True)
    
    return pts_db

    
def get_timeseries_for_pts_multicell(out_dir, spec_index, start_yr, end_yr, img_dir, image_type, grid_file, cell_list,
                            ground_polys, npts, apply_masks, seed, load_samp, ptfile):
    '''
    Returns datetime dataframe of values for sampled pts (n={'npts}) for each polygon in {'polys'}
    OR for previously generated points with {load_samp}=True and {pt_file}=path to .csv file
     (.csv file needs 'XCoord' and 'YCoord' fields (in this case, groundpolys, oldest, newest, npts and seed are not used))
        if {ptfile} is "centroids" instead of path, will get coords from centroid of each grid cell in {cell_list}. 
    for all images of {image_type} acquired between {'start_yr'} and {'end_yr'} in {'TS_Directory'}
    imageType can be "Planet" 'Sentinel', 'Landsat', or 'All'
    Output format is a datetime object with date (YYYY-MM-DD) on each row and sample name (polygonID_pt#) in columns
    '''

    allpts = pd.DataFrame()

    for cell in cell_list:
        df = gpd.read_file(grid_file)
        gbds = df.query(f'UNQ == {cell}').geometry.total_bounds
        #cell_bbox = box(bb[0],bb[1],bb[2],bb[3])
        #grid_bounds = gpd.GeoDataFrame(gpd.GeoSeries(grid_bbox), columns=['geometry'], crs=crs_grid)
        #print(f'cell bounds = {gbds}')
    
        ts_stack = []
        print ('working on cell {}'.format(cell))
        if load_samp == True:
            points = get_pts_in_grid (grid_file, cell, ptfile)
            print(points)
            polys = None
        else:
            polys = get_polygons_in_grid (grid_file, cell, ground_polys)
            points = None
        if isinstance(points, gpd.GeoDataFrame) or polys is not None:
            if image_type == 'Planet':
                for img in os.listdir(img_dir):
                    if img.endswith('.tif') and 'udm' not in img and img.startswith('2'):
                        img_yr, img_doy = get_img_date(img, image_type, data_source=None)
                        if img_yr >= start_yr and img_yr <= end_yr:
                            with rio.open(os.path.join(img_dir,img)) as src:
                                bds = src.bounds
                                #print(f'image bounds = {bds}')
                                if (gbds[2] < bds[0]) or (gbds[0] > bds[2]) or (gbds[1] > bds[3]) or (gbds[3] < bds[1]):
                                    continue
                                else:
                                    print(f'found overlap with {img}')
                                    ts_stack.append(os.path.join(img_dir,img))

        if ts_stack and len(ts_stack) > 0:    
            ts_stack.sort()
            if load_samp == True:
                polys=None
                pts = get_index_vals_at_pts(out_dir, ts_stack, image_type, polys, spec_index, npts, apply_masks, seed=88, load_samp=True, ptgdb=points)
            else:
                pts = get_index_vals_at_pts(out_dir, ts_stack, image_type, polys, spec_index, npts, apply_masks, seed=88, load_samp=False, ptgdb=None)
            start = pts.shape[0]
            
            allpts = pd.concat([allpts, pts])

        else:
            print('skipping this cell')
            pass

    ts = allpts.transpose()
    ts['date'] = [pd.to_datetime(e[:4]) + pd.to_timedelta(int(e[4:]) - 1, unit='D') for e in ts.index]
    ##Note if mask is applies, columns will be object. Need to change to numeric or any NA will result in  NA in average.
    #print(ts.dtypes)
    cols = ts.columns[ts.dtypes.eq('object')]
    for c in cols:
        ts[c] = ts[c].astype(float)
    #print(TS.dtypes))
    ts.set_index('date', drop=True, inplace=True)
    ts=ts.sort_index()

    ts['ALL'] = ts.mean(axis=1)
    ts['stdv'] = ts.std(axis=1)
    
    pd.DataFrame.to_csv(ts, os.path.join(out_dir,'TS_{}_{}-{}.csv'.format(spec_index, start_yr, end_yr)), sep=',', na_rep='NaN', index=True)
    
    return ts
    

def load_TS_from_file(ts_file):
    ts = pd.read_csv(ts_file)
    ts.set_index('date', drop=True, inplace=True)
    ts.index = pd.to_datetime(ts.index)
    ts = ts.sort_index()

    return ts

def convert_timeseries_to_monthly_doy(full_ts_dir,mo,newdir):
    '''
    Splits full time-series dataframe into monthly chunks for easier comparison with Planet(which are already
    chunked in processing). Note output index is now 'imDay' with format YYDDD
    '''
    for f in os.listdir(full_ts_dir):
        if f.endswith('.csv'):
            df = pd.read_csv(os.path.join(full_ts_dir,f))
            df.set_index('date', drop=True, inplace=True)
            df.index = pd.to_datetime(df.index)
            dfmo = df[df.index.month == int(mo)]
            dfmo.index = dfmo.index.strftime('%Y%j')
            dfmo.index.names = ['imDay']
            new_name = mo+f[4:]
            print(new_name)
            pd.DataFrame.to_csv(dfmo, os.path.join(full_ts_dir,newdir,new_name), sep=',', na_rep='NaN', index=True)

def convert_timeseries_to_doy(tsdf, tyr, start_day=0, end_day=365, season=True):
    new_df = tsdf[tsdf.index.year == tyr]
    #change index to just month-day:
    new_df.index = new_df.index.dayofyear
    new_df = new_df[new_df.index >= start_day]
    new_df = new_df[new_df.index <= end_day]
    if season==True:
        new_df.index = (new_df.index - start_day) + 1

    return new_df

def get_image_diffs(ts):
    '''
    takes a time series and returns difference matrix for V(t+1)-V(t) where t is the 
    '''
    ts_f = ts.ffill(axis=0)
    ts_t2 = ts_f.copy()
    ts_t2['up_one']=ts_t2.index - 1
    ts_t2.set_index(ts_t2['up_one'], inplace=True)
    diff = ts_t2 - ts_f
    diff.drop(columns=['up_one'],inplace=True)
    
    return diff

def get_cloud_mask(mask_file, mask_type):
    print('masking image')
    '''
    For PlanetScope, gets cloud mask from usable data mask file. Returns 3 different mask options.
    if {mask_type} = 'udm2all': 
        returns the unclear mask that is the sum of all mask bands in the udm2 file (also on band 1)
    if {mask_type} = 'udm2cs':
        returns the cloud/shadow mask only
    if {mask_type} = 'udm2csorig':
         returns the cloud/shadow mask (bands 3 & 6) + the original udm mask (band 8)
    if {mask_type} = 'blackfill':
         returns area (outside of actual imagery. (TODO: may need to add other masks from above))
    Note in udm2file:
        for band 1: clear =1, notclear = 0, blackfill = 0.
        for cloud and shadow bands: notclear = 1. clear =0, but backfill also =0.
    '''
    if 'udm2' in mask_file:
        with rio.open(mask_file, 'r') as src:
            udm2_array = src.read(1)
            orig_udm = src.read(8)
            cloud_band = src.read(6)
            shadow_band = src.read(3)

            orig_mask =  orig_udm != 0
            cloud_mask = cloud_band != 0
            shadow_mask = shadow_band != 0
                
            if mask_type == 'udm2all':
                mask_out = udm2_array != 0
            elif mask_type == 'udm2cs':
                mask_out = cloud_mask | shadow_mask
            elif mask_type == 'udm2csorig':
                mask_out = cloud_mask | shadow_mask | orig_mask
            elif mask_type == 'blackfill':    
                mask_out = orig_udm == 1
            else:
                print (f'{mask_type} is not currently an out_type option, Choose udm2all, utm2cs, or udm2csorig')
    else:
        print (f'{mask_file} is not a udm2 file, skipping udm2 calcs')
        
    return mask_out

def mask_clouds(img_file, mask_type):
    basename = str(os.path.basename(img_file)[:21]) #for PlanetScope the first 21 digits are always unique???
    for f in os.listdir(Path(img_file).parent):
        if basename in f and f.endswith('udm2.tif'):
            print(f'found a mask!: {f}')
            mask_file = os.path.join(Path(img_file).parent,f)
    try:
        msk = get_cloud_mask(mask_file, mask_type)
        masked = True
    except:
        print(f'problem masking {img_file}')
        masked = False
        
    if masked == True:     
        with xr.open_rasterio(img_file) as img:      
            img_masked = img.where(msk != 1, other=np.nan)
    else:
        with xr.open_rasterio(img_file) as img:
            img_masked = img
   
    return img_masked

def normalize(array):
    array_min, array_max = array.min(), array.max()
    return (array - array_min) / (array_max - array_min)

def gammacorr(band, gamma):
    return np.power(band, 1/gamma)

def plot_img(raster, gamma, bandcombo):
    '''
    use bandcombo = '321' for truecolor, 
    bandcombo='431' for Planet false ir
    bandcombo = '643' for Sentinel-2 false ir
    '''    
    with rio.open(raster) as src:
        red0 = src.read(int(bandcombo[0]), masked=True)
        green0 = src.read(int(bandcombo[1]), masked=True)
        blue0 = src.read(int(bandcombo[2]), masked=True)
        red = np.ma.array(red0, mask=np.isnan(red0))
        red[red < 0] = 0
        green = np.ma.array(green0, mask=np.isnan(green0))
        green[green < 0] = 0
        blue = np.ma.array(blue0, mask=np.isnan(blue0))
        blue[blue < 0] = 0
        
    # gamma correct and normalize:
    red_g=gammacorr(red, gamma)
    blue_g=gammacorr(blue, gamma)
    green_g=gammacorr(green, gamma)

    red_n = normalize(red_g)
    green_n = normalize(green_g)
    blue_n = normalize(blue_g)

    # Stack bands
    rgb = np.dstack([red_n, green_n, blue_n])
    
    return rgb           
        
def ts_to_obs_chart(tsdf, labels, print_df=False, out_dir=None, series_name=None):
    tsdf2 = tsdf.drop(['ALL', 'stdv'], axis=1)
    # change all values to 1 (valid obs) or 0 (no obs):
    tsdf2 = tsdf2.notnull().astype('int')
    for inx, column in enumerate(tsdf2):
        tsdf2[column] = (inx + 1) * tsdf2[column]
    # rename columns to match cell list (pt labels might have been changed):
    tsdf2.set_axis(labels, axis=1,copy=False)
    if print_df == True:
        pd.DataFrame.to_csv(tsdf2, os.path.join(out_dir,'obs_{}.csv'.format(series_name)), sep=',', na_rep='NaN', index=True)
    return tsdf2