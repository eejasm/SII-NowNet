import numpy as np
import pandas as pd
import math

def generate_tiles(data, no_x_grids, no_y_grids, target_grid_pixel_res, alpha):
    
    """
    
    This function produces the input data for each prediction grid. The output is a stack of 2d fields (tiles). 
    
    data: the himawari BT or orography data that covers the whole prediction domain
    no_x_grids: number of prediction grids in the x direction
    no_y_grids: number of prediction grids in the y direction
    target_grid_pixel_res: the number of himawari BT pixels in the x or y direction (must be the same)
    alpha: sets the size of the input data field. alpha = 1 produces a 3x3 grid around the prediction grid, alpha = 2 produces a 5x5 grid around the prediction grid etc. Default alpha=1.
    
    """
    
    output_tile = []
    
    field = np.zeros((data.shape[0], data.shape[1]))
    
    x_length = target_grid_pixel_res
    y_length = target_grid_pixel_res
    
    # can't include the outermost prediction grids because the input data for those tiles stretches outside the domain
    for i in range(1, int(no_x_grids)-1):
         
        for j in range(1, int(no_y_grids)-1):
                               
            
            # use the coordinates to find the equivalent upper and lower coordinates in the much higher res predictor fields
            lower_x = math.ceil(float((i/no_x_grids)*data.shape[0]))
            lower_y = math.ceil(float((j/no_y_grids)*data.shape[1]))
            upper_x = lower_x + x_length
            upper_y = lower_y + y_length
    
                
            ## add additonal length to dimensions as specified by alpha
            lower_x = lower_x - alpha*x_length
            upper_x = upper_x + alpha*x_length
            lower_y = lower_y - alpha*y_length
            upper_y = upper_y + alpha*y_length
            
            cropped_img_tile = data[lower_x:upper_x, lower_y:upper_y]
            output_tile.append(cropped_img_tile)
    
    tiles = np.stack(output_tile)
    
    return tiles
