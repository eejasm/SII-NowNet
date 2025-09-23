import pandas as pd
import datetime
from datetime import datetime, timedelta
import numpy as np
import math
import numpy.ma as ma
import xarray as xr
import os

from scipy.ndimage.filters import uniform_filter

from pyproj import Transformer
import glob
from satpy import Scene
from glob import glob

def normalize(array, max_val, min_val):
    
    array = (array - min_val) / (max_val - min_val + 1e-13)
    
    return array

def load_data(time):
    
    # Use fast single glob + in-memory filtering
    base_path = f'/gws/nopw/j04/swift/WISER-EWSA/HRIT/data/{time[:-4]}'
    all_files = glob(os.path.join(base_path, f'*{time}*'))

    ir_108_filenames = [f for f in all_files if 'IR_108' in f]
    epi_filenames    = [f for f in all_files if 'EPI' in f]
    pro_filenames    = [f for f in all_files if 'PRO' in f]
    filenames = ir_108_filenames + epi_filenames + pro_filenames

    # Initialize and load only needed channel
    scene = Scene(reader='seviri_l1b_hrit', filenames=filenames)
    scene.load(['IR_108'])

    IR_108 = scene["IR_108"]

    # Projection and coordinate transform
    msg_proj = ccrs.Geostationary(central_longitude=0.0, satellite_height=35785831)
    transformer = Transformer.from_crs("EPSG:4326", msg_proj, always_xy=True)

    # set you lat/lon centre coords here
    lon, lat = 28.0, -26.2
    cen_x, cen_y = transformer.transform(lon, lat)

    # Nearest grid point
    cen_x = IR_108.sel(x=cen_x, y=cen_y, method='nearest').x.item()
    cen_y = IR_108.sel(x=cen_x, y=cen_y, method='nearest').y.item()

    # Find pixel index (faster than meshgrid)
    x_index = np.argmin(np.abs(IR_108.x.values - cen_x))
    y_index = np.argmin(np.abs(IR_108.y.values - cen_y))

    # Crop region
    south_africa_ir = IR_108.isel(x=slice(x_index - 208, x_index + 208), y=slice(y_index - 208, y_index + 208))

    return np.fliplr(south_africa_ir.values)

# intensification identification methodology
def generate_convection_intensification_fields(data,
                                               change_thresh, 
                                               coverage_thresh, 
                                               bt_thresh, 
                                               target_domain_shape):

    # generate the intensification target fields 
    
    bt_t1 = data[:,:,0]
    bt_t0 = data[:,:,1]
    bt_change = bt_t1 - bt_t0

    # BT pixles must get colder than 20 K and finish below 235 K to be considered intensifying
    binary_bt_change = np.logical_and(bt_change>=change_thresh, bt_t0<=bt_thresh).astype(int)

    # rescale to 26x26 grid - for each grid check whether the fraction of intensifying pixels is greater than 5%
    scaled_convection_target, grid_coverage = rescale(binary_bt_change, target_domain_shape, coverage_thresh)
    
    return scaled_convection_target

# intensification identification methodology
def generate_convection_initiation_fields(data, 
                                          change_thresh, 
                                          coverage_thresh, 
                                          bt_thresh, 
                                          kernel_size, 
                                          smooth_thresh,
                                          target_domain_shape):
            
    
    bt_t1 = data[:,:,0]
    bt_t0 = data[:,:,1] 

    # threshold T-1 field at 255 K
    binary_bt_t1 = np.where(bt_t1<=bt_thresh+change_thresh,1,0)
    # threshold T-0 field at 235 K
    binary_bt_t0 = np.where(bt_t0<=bt_thresh,1,0)

    # smooth the T-1 thresholded field the simulate the growth/propagation of the presnet convection
    smooth_bt_t1 = uniform_filter(binary_bt_t1, size=kernel_size, mode="constant", output = float, cval=0.0)
    smooth_bt_t1 = np.where(smooth_bt_t1<smooth_thresh,0,1)

    # remove the growing/propgagting present convection from the T-0 field, to leave only the newly initiating convection
    convection_initiation = binary_bt_t0-smooth_bt_t1
    binary_convection_initiation = np.where(convection_initiation<=0,0,1)

    # rescale to 26x26 grid - for each grid check whether the fraction of initiating pixels is greater than 5%
    scaled_binary_initiation, grid_coverage = rescale(binary_convection_initiation, target_domain_shape, coverage_thresh)
    
    return scaled_binary_initiation

def rescale(image, target_domain_shape, thresh):

    # rescales the initiation/intensification grids to 26x26 and adds the >5% check in 
    
    x_dims = int(image.shape[0]/target_domain_shape[0])
    y_dims = int(image.shape[1]/target_domain_shape[1])
    
    scaled_image = np.zeros((target_domain_shape[0], target_domain_shape[1]))
    grid_coverage = np.zeros((target_domain_shape[0], target_domain_shape[1]))

    for i in range(target_domain_shape[0]):
        
        for j in range(target_domain_shape[1]):
            
            grid_cov = np.sum(image[i*x_dims:(i+1)*x_dims, j*y_dims:(j+1)*y_dims])
            grid_coverage[i,j] = grid_cov
            
            if grid_cov >= (thresh*(x_dims**2)):
                scaled_image[i,j] = 1
    
    return scaled_image, grid_coverage

def prepare_target(input_data, init_or_intens):
        
    if init_or_intens == 'intensification':
    
        target = generate_convection_intensification_fields(input_data,
                                                                       20, 
                                                                       0.05, 
                                                                       235, 
                                                                       (26,26))
    if init_or_intens == 'initiation':

        target = generate_convection_initiation_fields(input_data, 
                                                            20, 
                                                            0.05, 
                                                            235, 
                                                            20, 
                                                            0.05,
                                                            (26,26))
    return target