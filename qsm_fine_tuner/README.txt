# List of paths to check before running the optimizer codes:

I. Check the path of you GM and WM mask, they need to be compatible with your input images, this is crucial as this is used for the metric calculation.
II. Check iter_folder and txt_file_path inside the configure_experiment_run function, they need to point to the same folder, this is were the results are saved during optimization.
III. Check that you are using the correct ground truth according to the experiment you are running.
IV. Then, inside the optimizer function itself, checking the fm/lf path, header path and mask for processing path!
*Some algorithms may need more inputs than these 3 basic aforementioned, (MEDI, FANSI, ...)