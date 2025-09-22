# SII-NowNet
A repository for Simple Initiation and Intensification Nowcasting Neural Network (SII-NowNet) code


## 1. Basic guide

This directory goes provides a step-by-step guide of how to apply SII-NowNet. It provides so example data (for Sumatra and Java), along with the SII-NowNet models for 1 and 2 hour lead time. However, this specific example is for a 1 hour lead time nowcast. 

The guide is given in a jupyter notebook called SII-NowNet_example_guide.ipynb. The SII-NowNet models (for initiation and intensification) are kept within the tools directory. 

The tools directory also contains normalisation values (for processing the input to SII-NowNet) and some example data to run through.

predictor_data_generator.py is a script that extracts the predictor data for each prediction grid in the chosen domain.
target_data_generator.py is a script that produces the observations for verification (only possible if looking at historical cases)
utils.py contains fucntions for normalising the data before it goes into SII-NowNet.


## 2. SII-NowNet architecture

This is a jupyter notebook containing the basic form of the CNN underlying SII-NowNet. Currently it takes a 63x63x3 matrix as input, but this can be modified and re-trained on data wth different dimensions (to account for different domains) 

## 3. SII-NowNet_2

This directory contains the updated version of SII-NowNet, with a basic guide on how to use it. The three .ipynb guides are focussed on South Africa, Southern Africa and Indonesia. The Southern Africa and Indonesia domains use a tiling approach, which stitches together multiple SII-NowNet nowcasts into one whole domain (SII-NowNet is restricted to a 416x416 input), using smoothing at the tile overlaps. The South Africa domain uses just one SII-NowNet nowcast.

Before running these guides you must add a directory labelled './SII-NowNet_2_models/'. This directory should contain 2xinitiation models (1 and 2 hour lead times) and 3xintensification models (1, 2 and 3 hour lead times). These models called from a their own directories, which should be labelled as '1_initiation', '2_initiation', '1_intensification', '2_intensification', '3_intensification'. These directories can be found at https://drive.google.com/drive/folders/1Xucu1GsgaYnkYuVPbZmIhyIdAMXsY3wy. 

Your own Brightness Temperature data also needs to be added to this directory. A description of this data is within the guide. The shape each BT numpy field must be (416, 416).

SA_SIINowNet.py also been added to the directory. This is the .py script that is being run by the South African Weather Service (SAWS) to generate SII-NowNet nowcasts for the Southern Africa domain.

