import numpy as np
import pandas as pd
import datetime
from datetime import datetime, timedelta
from scipy.ndimage.filters import uniform_filter

def rescale(image, target_domain_shape, thresh):
    
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


def generate_convection_intensification_fields(data, time, change_thresh, coverage_thresh, t1_bt_thresh, target_domain_shape):
    
    convection_df = pd.DataFrame(columns=['time_of_initiation', 'scaled_convection_target', 'grid_coverage'])
    
    bt_t0 = data[0]
    bt_t1 = data[1]
    bt_change = data[0] - data[1]
    
    binary_bt_change = np.logical_and(bt_change>=change_thresh, bt_t1<=t1_bt_thresh).astype(int)
    
    scaled_convection_target, grid_coverage = rescale(binary_bt_change, target_domain_shape, coverage_thresh)
    
    
    convection_df = pd.concat([convection_df, pd.DataFrame({'time_of_initiation': time,                               
                                                            'scaled_convection_target': [scaled_convection_target],
                                                            'grid_coverage': [grid_coverage]})], ignore_index=True)
        
    
    return convection_df


def generate_convection_initiation_fields(data, time, change_thresh, coverage_thresh, t1_bt_thresh, kernel_size, smooth_thresh, target_domain_shape):
    
    convection_df = pd.DataFrame(columns=['time_of_initiation', 'scaled_convection_target', 'grid_coverage'])
        
    bt_t0 = data[0]
    bt_t1 = data[1]
    
    binary_bt_t0 = np.where(bt_t0<=t1_bt_thresh+change_thresh,1,0)
    binary_bt_t1 = np.where(bt_t1<=t1_bt_thresh,1,0)
    
    smooth_bt_0 = uniform_filter(binary_bt_t0, size=kernel_size, mode="constant", output = float, cval=0.0)
    smooth_bt_0 = np.where(smooth_bt_0<smooth_thresh,0,1)
    
    convection_initiation = binary_bt_t1-smooth_bt_0
    binary_convection_initiation = np.where(convection_initiation<=0,0,1)
    
    scaled_binary_initiation, grid_coverage = rescale(binary_convection_initiation, target_domain_shape, coverage_thresh)
    
    convection_df = pd.concat([convection_df, pd.DataFrame({'time_of_initiation': time,
                                                            'scaled_convection_target': [scaled_binary_initiation],
                                                            'grid_coverage': [grid_coverage]})], ignore_index=True)
    
    
    return convection_df


def intensification_target_generator(data, 
                                     time,
                                     change_thresh, 
                                     coverage_thresh,
                                     t1_bt_thresh,
                                     target_grid_pixel_res):
    
    target_grid_shape = (target_grid_pixel_res, target_grid_pixel_res)

    x_length = data.shape[1]
    y_length = data.shape[2]
    
    target_domain_shape_x = x_length/target_grid_pixel_res
    target_domain_shape_y = y_length/target_grid_pixel_res
    
    if (target_domain_shape_x != int(target_domain_shape_x)) or (target_domain_shape_y != int(target_domain_shape_y)):
        
        raise Exception('Target grid pixel resolution must be a multiple of input data dimension')
        
    
    target_domain = np.ones((int(target_domain_shape_x), int(target_domain_shape_y)))
    target_domain_shape = target_domain.shape
    
    #print(target_domain_shape[0])
    
    df = generate_convection_intensification_fields(data, 
                                                    time, 
                                                    change_thresh, 
                                                    coverage_thresh, 
                                                    t1_bt_thresh, 
                                                    target_domain_shape)
    

    return df   


def initiation_target_generator(data, 
                                time, 
                                change_thresh, 
                                coverage_thresh, 
                                t1_bt_thresh, 
                                kernel_size, 
                                smooth_thresh, 
                                target_grid_pixel_res):
    
    target_grid_shape = (target_grid_pixel_res, target_grid_pixel_res)

    x_length = data.shape[1]
    y_length = data.shape[2]
    
    target_domain_shape_x = x_length/target_grid_pixel_res
    target_domain_shape_y = y_length/target_grid_pixel_res
    
    if (target_domain_shape_x != int(target_domain_shape_x)) or (target_domain_shape_y != int(target_domain_shape_y)):
        
        raise Exception('Target grid pixel resolution must be a multiple of input data dimension')
    

    
    target_domain = np.ones((int(target_domain_shape_x), int(target_domain_shape_y)))
    target_domain_shape = target_domain.shape
    
    
    df = generate_convection_initiation_fields(data, 
                                               time, 
                                               change_thresh, 
                                               coverage_thresh, 
                                               t1_bt_thresh, 
                                               kernel_size, 
                                               smooth_thresh, 
                                               target_domain_shape)


    return df   