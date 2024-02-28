# RFI
The link to download the DenseNet pre-trained model on the simulated metal bridge is the following:
https://drive.google.com/file/d/1HM1_c58WfOD4qRsL4cqemCstMavxJmgn/view?usp=drive_link
The MultiImgTrain2.py shows the training pipeline code.

The code in signal_to_img2.py. which performs the conversion of the 1D signals to the corresponding CWT image, use a modified version of the ssqueezepy package.

First, the package needs to be installed with:

pip install ssqueezepy

Then, the imshow function of the ssqueezepy library must me modified as follows:

In the file ssqueezepy/visuals.py substitute the actual imshow function definition with the imshow function defined into modified_imshow.py file 
