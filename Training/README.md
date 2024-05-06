These scripts can be run with Python, IPython or in a Jupyter notebook.

Please specify the directory to which the training files and the configuration file should be saved to before starting training.
To change the directory, just overwrite `save_root` and `save_name_config` under `# %% Save directory` within the script.

Please also specify the location of the python executable, that has the framework installed, if you want to start training from within the script (default).
To change the directory, just overwrite `python_loc` under `# Start training` or `# %% Start training`

Training is started from within the script (see code block `# Start training` or `# %% Start training`.
To start training from a console, just run the code to create the config file first and use this config file to start training. 
Using the tag `--show-example --napari` allows to get batches of data that can be viewed with napari.

Feel free to adjust parameters or settings by changing the values in the script or specify changes when starting training from a console with a tag.
For example, for a test run, the number of epochs can be set to 50 with the tag `--epochs 50`.

## Experiments
*   [UNET_2D](UNET_2D.py) - Training a 2D network (instead of 3D)
*   [UNET_24x80x80](UNET_24x80x80.py) - Using an input size of 24x80x80 (instead of 24x200x200)
*   [UNET_Without_Norm](UNET_Without_Norm.py) - No normalization layers
*   [UNET_BN](UNET_BN.py) - Using batch normalization
*   [UNET_IN](UNET_IN.py) - Using instance normalization
*   [UNET_CE](UNET_CE.py) - Using weighted cross-entropy as loss function
*   [UNET_DICE](UNET_DICE.py) - Using weighted dice loss as loss function
*   [UNET_MSE](UNET_MSE.py) - Using MSE loss as loss function
*   [UNET_Dropout_20](UNET_Dropout_20.py) - Using dropout layers (20%) instead of normalization layers
*   [UNET_RELU](UNET_RELU.py) - Using ReLU insted of ELU
*   [UNET_Leaky_Relu](UNET_Leaky_Relu.py) - Using leaky ReLU instead of ELU
*   [UNET_Levels](UNET_Levels.py) - Training a U-Net with different depths (Level: 2, 3, 4)
*   [UNET_Cross-validation](UNET_Cross-validation.py) - 6-fold cross-validation
*   [UNET_Final](UNET_Final.py) - Training the final model (5-times, different seeds)
*   [UNET_Binary_MT](UNET_Binary_MT.py) - Training a binary classifier with BG and MT
*   [UNET_Binary_CM](UNET_Binary_CM.py) - Training a binary classifier with BG and CM

## Example
Running the script after changing the directories:

    .../python .../UNET_DICE.py

Creating the config file and running from console:

    .../python .../train.py .../config_file.py --epochs 50 --update 25 --use-cache

