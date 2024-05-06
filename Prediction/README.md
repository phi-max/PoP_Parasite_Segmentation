Contains the scripts to perform inference. It also allows to examine the validation results with napari.
Please specify the directory where the training results of an experiment are saved/should be saved to.
To change the directory, just overwrite `dir` under `# %% Directory` within the script.

## How to use the scripts

The code is structured in code blocks, marked with `# %%`.
To use napari, run the code blocks in IPython or in a Jupyter Notebook.
After starting a IPython kernel, please run `%gui qt` to make sure that napari will work. 


[Prediction_validation_data](Prediction_validation_data.py):
*   Run the code under `# %% Imports`
*   Specify the directory, in which the training results of a model are located and run the line under `# %% Directory`
*   To perform inference on the validation data, execute the code under `# %% Predict validation data (Argmax)`
    * A folder with the name `predict` within the experiment folder is created (if it does not exist already) in which the predictions are saved 
    * If you do not want to save the predictions on the hard drive uncomment the `imsave()` lines in the `# Prediction loop`(3 lines)
*   To simply load the prediction(s) of the validation data, run the code under `# %% Load predictions validation data`
*   To visualize the results with napari, run the code block `# %% Visualize with napari`
*   To get the best_model's statistics (DSC, DSC per class, epoch) on the validation data, run `# %% Statistics validation data`
*   To plot the training progress (as illustrated in the thesis), run `# %% Statistics plot (Loss, learning rate, DSC mean, DSC per class)`
*   To output probability maps instead of segmentation maps, run `# %% Predict validation data (Probabilities)`
*   To visualize the probability maps, run `# %% Visualize with napari (Probabilities)` afterwards

[Prediction_unseen_data](Prediction_unseen_data.py):
*   Run the code under `# %% Imports`
*   Specify the directory, in which the training results of a model are located and run the line under `# %% Directory`
*   To perform inference on the validation data, execute the code under `# %% Predict large images (GPU, Argmax)`
    * A folder with the name `predict` within the experiment folder is created (if it does not exist already) in which the predictions are saved 
    * If you do not want to save the predictions on the hard drive uncomment the `imsave()` lines in the `# Prediction loop`(3 lines)
*   To simply load the prediction(s) of the validation data, run the code under `# %% Load predictions`
*   To visualize the results with napari, run the code block `# %% Visualize with napari`
    * If more than 1 prediction is available, you can press `n` in napari to get the next prediction
    * You can switch between a 2D and 3D mode
*   To output probability maps instead of segmentation maps, run `# %% Predict large images (Probabilities)`
*   To visualize the probability maps, run `# %% Visualize with napari (Probabilities)` afterwards

[Prediction_extra](Prediction_extra):
*   Contains additional scripts