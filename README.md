# Applying a U-Net to segment microtubules in electron tomograms of *Trypanosoma brucei*

This repository contains the python files for creating, training and testing the models.
It also contains the code for creating the plots that were used in this thesis.

*   [Training scripts](Training) - Contains the code to train the models 
*   [Prediction scripts](Prediction) - Contains the scripts that can be used to predict the validation volumes (or any other volume)
*   [Plots & other scripts](Plots%20&%20other%20scripts) - Contains the code to create the plots (e.g. activation functions, etc.) and additional scripts (e.g. computing the FOV etc.)

If you want to reproduce training, take a look at the [training scripts](Training).
If you want to use the trained models to predict the segmentation on unseen data or the validation data, take a look at the [prediction scripts](Prediction).
If you just want to visualize the segmentation results (with napari),
check out [this example](EXAMPLE.md).
Alternatively, you can take a look at the respective parts within the [prediction scripts](Prediction).
