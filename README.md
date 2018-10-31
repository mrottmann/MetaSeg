
## What is MetaSeg:

MetaSeg is a post-processing tool for semantic segmentation neural networks. For each component/object in the segmentation, MetaSeg on the one hand provides a method that predicts whether this particular component intersects with the ground truth or not. This task can be understood as meta classifying between the two classes {IoU=0} and {IoU>0}. On the other hand MetaSeg also provides a method for quantifying the uncertainty of each predicted segment by predicting IoU values via regression. MetaSeg is a method that treats the neural network like a blackbox, i.e., at inference time it only uses the softmax output of the neural network.

For further reading and also in case you want to re-use the code and publish results, please refer to http://arxiv.org/XXXXXXX.


## Preparation:

We assume that the user is already using a neural network for semantic segmentation and a corresponding dataset. For each image from the segmentation dataset, MetaSeg requires a hdf5 file that contains the following data:

- a three-dimensional numpy array (image dimensions times number of classes that the segmentation network can predict) that contains the softmax probabilities computed for the current image
- the filename of the current input image
- a two-dimensional numpy array (of input image dimension) that contains the ground truth class indices for the current image

MetaSeg provides a function "probs_gt_save" in "metaseg_io.py" to store this information for each image. Before running MetaSeg, please edit all necessary paths stored in "global_defs.py". You can overwrite your default values stored in "global_defs.py" temporarily by setting command line arguments in "metaseg_eval.sh", which can be used for running MetaSeg. MetaSeg is CPU based and parts of MetaSeg trivially parallize over the number of input images, adjust "NUM_CORES" in "global_defs.py" or "metaseg_eval.sh" to make use of this.


## Deeplabv3+ and Cityscapes:

The results in http://arxiv.org/XXXXXXX have been obtained from two Deeplabv3+ networks (https://github.com/tensorflow/models/tree/master/research/deeplab) together with the Cityscapes dataset (https://www.cityscapes-dataset.com/). For using the latter you need to enroll at Cityscapes on the website. For details on using Deeplabv3+ networks in combination with cityscapes, we refer to the README page https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/cityscapes.md.

 ## Packages and their versions we used:

- tensorflow-gpu==1.9.0
- pandas==0.22.0
- scikit-learn==0.19.0
- Pillow==4.2.1
- matplotlib==2.0.2
- scipy==0.19.1
- h5py==2.7.0
- Python==3.4.6
- Cython==0.27
- gcc==4.8.5

## Authors:
Matthias Rottmann (University of Wuppertal), Thomas Paul Hack (Leipzig University), Pascal Colling (University of Wuppertal).
