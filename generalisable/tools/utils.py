import numpy as np

def normalize(array, max_val, min_val):
    
    array = (array - min_val) / (max_val - min_val + 1e-13)

    return array

def prepare_test_input(input_data):
    
    """
    
    This function normalizes the data between pre-set limits. These limits can be found in the normalisation_values directory.
    
    input_data: an array of shape (3 x N x len_x x len_y). 3 refers to the number of predictor channel - channel 0:BT, channel 1:BT change, channel 2:orography. N is the number of prediction grids, len_x and len_y are the 
    number of pixels in the x and y direction
    
    Output is a normalized array of shape (N x len_x x len_y x 3)

    
    """
    abs_him_max, abs_him_min = np.load('./tools/normalisation_values/abs_him_max.npy'), np.load('./tools/normalisation_values/abs_him_min.npy')
    diff_him_max, diff_him_min = np.load('./tools/normalisation_values/diff_him_max.npy'), np.load('./tools/normalisation_values/diff_him_min.npy') 
    orog_max, orog_min = np.load('./tools/normalisation_values/abs_orog_max.npy'), np.load('./tools/normalisation_values/abs_orog_min.npy')
    
    abs_him = normalize(input_data[0], abs_him_max, abs_him_min)
    diff_him = normalize(input_data[1], diff_him_max, diff_him_min)
    orog = normalize(input_data[2], orog_max, orog_min)
    
    norm_input = np.stack([abs_him, diff_him, orog], axis=-1)
    
    return norm_input