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

This is a jupyter notebook containing the basic form of the CNN underlying SII-NowNet. Currentlt it takes a 63x63x3 matrix as input, but this can be modified and re-trained on data wth different dimensions (to account for different domains) 
