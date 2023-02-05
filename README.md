README
===========================

### Author
Yixuan Shao

Contact: shaoyixuan013@gmail.com

### Project description
In this project, I realize a hyperspectral image reconstruction algorithm with high spatial and spectral resolution from a single shot. By placing a dispersive optic in front of an conventional 3-channel camera, the PSF would be horizontally dispersive and there would be dispersion at edges. With the prior knowledge of small total variation and spectral smoothness, reconstructing the hyperspectral image from dispersion at the edges is formed as a 3-step inverse problem. Solving the inverse problem using the conjugate gradient method and alternating direction method of multipliers, I reconstruct the spectra from 430 to 650 nm with 10 nm intervals.

The detail of the algorithm and result analysis is in the report.pdf file.

### Usage
I provide three examples to run. They reconstruct the hyperspectral image of the input of lemons, a ColorChecker, and a toy dog, respectively. Simply run the file  example_lemon.py  example_colorchecker.py  example_dog.py in Python. The results will be stored in result_lemon, result_colorchecker, result_dog folder respectively. 

I also analyze how the noise level impact the result. Run noise_analysis.py for noise analysis. 

You can also run parameter_analysis.py to analyze the impact of different choices of optimization parameters on the result.

### Helper functions
forward.py provides functions for the dispersive image formation model.

edgeOperator.py provides functions for finding edges and manipulating edges.

step1.py  step2.py and step3.py include functions of the three core steps to reconstruct the hyperspectral image.

finite_differences.py includes operators for finding spatial gradient, spectral gradient and their conjugate operators.

evaluate_model.py provides the function to evaluate the spectral RMSRE (root mean squared relative error).

### Datasets
The folder fake_and_real_lemon_slices_ms and stuffed_toys_ms are the hyperspectral image datasets. They include the groundtruth of the hyperspectral images. 

The folder stuffed_toys_ms is downloaded from a public hyperspectral image dataset https://www1.cs.columbia.edu/CAVE/databases/multispectral/stuff/

The folder fake_and_real_lemon_slices_ms is downloaded from https://www1.cs.columbia.edu/CAVE/databases/multispectral/real_and_fake/

xyzbar.mat contains the matrix used to convert hyperspectrum to XYZ color space based on CIE 1931.

### Citation
Please cite this github repo if you would like to use my code.
