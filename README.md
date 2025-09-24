# SII-NowNet
A repository for Simple Initiation and Intensification Nowcasting Neural Network (SII-NowNet) code

## 1. SII-NowNet_1 architecture

This is a jupyter notebook containing the basic form of the CNN underlying the first version of SII-NowNet. Currently it takes a 63x63x3 matrix as input, but this can be modified and re-trained on data wth different dimensions (to account for different domains) 

## 2. SII-NowNet_1

This directory goes provides a step-by-step guide of how to apply SII-NowNet. It provides so example data (for Sumatra and Java), along with the SII-NowNet models for 1 and 2 hour lead time. However, this specific example is for a 1 hour lead time nowcast. 

The guide is given in a jupyter notebook called SII-NowNet_example_guide.ipynb. The SII-NowNet models (for initiation and intensification) are kept within the tools directory. 

The tools directory also contains normalisation values (for processing the input to SII-NowNet) and some example data to run through.

predictor_data_generator.py is a script that extracts the predictor data for each prediction grid in the chosen domain.
target_data_generator.py is a script that produces the observations for verification (only possible if looking at historical cases)
utils.py contains fucntions for normalising the data before it goes into SII-NowNet.

## 3. SII-NowNet_2 architecture

This jupyter notebook contains the architecture for SII-NowNet_2. SII-NowNet_2 is an improved version, which produces greater skill over Sumatra (especially for initiation nowcasts). SII-NowNet_2 is based on a U-Net architecture with 4 encoder/decoder stages and an extra convolutional layer for final processing into the 26x26 nowcast grid. SII-NowNet_2 takes a 416x416x2 dimension input, which represents the BT fields (416x416 dimension) at T-0 and T-1 hours (2 dimension). 

## 4. SII-NowNet_2

This directory contains guides for using SII-NowNet_2. The three .ipynb guides are focussed on South Africa, Southern Africa and Indonesia. The Southern Africa and Indonesia domains use a tiling approach, which stitches together multiple SII-NowNet nowcasts into one whole domain (SII-NowNet is restricted to a 416x416 input), using smoothing at the tile overlaps. The 'tiling' approach used to form the larger SII-NowNet images differs between the two regions because the BT data is in a different projections. The South Africa domain uses just one SII-NowNet nowcast and therefore does not require any 'tiling' approach.

Before running these guides you must add a directory labelled './SII-NowNet_2_models/'. This directory should contain 2xinitiation models (1 and 2 hour lead times) and 3xintensification models (1, 2 and 3 hour lead times). These models called from a their own directories, which should be labelled as '1_initiation', '2_initiation', '1_intensification', '2_intensification', '3_intensification'. These directories can be found at https://drive.google.com/drive/folders/1Xucu1GsgaYnkYuVPbZmIhyIdAMXsY3wy. 

Your own Brightness Temperature data also needs to be added to this directory. A description of this data is within the guide. The shape each BT numpy field must be (416, 416).

SA_SIINowNet.py also been added to the directory. This is the .py script that is being run by the South African Weather Service (SAWS) to generate SII-NowNet nowcasts for the Southern Africa domain.

## 4. Target data generation
The code for the initiation/intensification methodologies (the target data) is provided in the generate_target_data.py scripts. These scripts require 2 BT fields, separated an hour apart, with 416x416 pixel dimension. From these images the initiation/intensification events are identified and scaled down to a 26x26 grid - this is the final target domain for SII-NowNet. 
