## How to use napari to visualize the results

napari should be installed in your conda environment.
Make sure to activate this env. 

    conda activate <env_name>

Start napari

    napari
    
Open the IPython console within napari (bottom left button) and enter the following lines:

    from skimage.io import imread
    
    img = imread('<input_file_location>')
    pred = imread('<prediction_file_location>')

![example_01](example_images/example_01.png)

Add the image and the prediction to napari by entering:

    viewer.add_image(img)
    viewer.add_labels(pred)

![example_02](example_images/example_02.png)

Alternatively, you can run napari from a IPython console

        python -m IPython
        
and enter the following lines:

        %gui qt
        import napari
        viewer = napari.Viewer()

![example_03](example_images/example_03.png)
