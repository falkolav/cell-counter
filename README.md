# CNN-based regressor and ML pipeline for Automatic Cell Counting
This is the implementation of two automated cell counting pipelines ("CNN based regressor and ML pipeline) as described in the following paper:

- Falko Lavitt, Demi J. Rijlaarsdam, Dennet van der Linden, Ewelina Weglarz-Tomczak, and Jakub M. Tomczak, _Automatic cell counting using a Convolutional Neural Network-based regressor with an application to microscope images of human cancer cell lines_, preprint, 2021
## Requirements
The code is compatible with:

- numpy~=1.18.1
- matplotlib~=3.2.1
- scikit-learn~=0.22.1
- pillow~=7.1.1
- xgboost~=1.0.2
- pandas~=1.0.3
- scikit-image~=0.16.2
- fastai~=2.2.5

## Data
The experiments can be run on the dataset described in the paper, which can be downloaded from https://doi.org/10.5281/zenodo.4428844.
In total, the dataset contains 165 labeled images of a human osteosarcoma cell line (U20S), a type of bone cancer and and a human leukemia cell line (HL-60), that we split into the training set (133 images) and the test set (32 images).



## Setup
1. Install the requirements specified in the requirements.txt file
2. Download the dataset from the link mentioned in the 'data' folder
3. Set-up your experiment using the experiment.py. file in either the cnn-regressor or ml-pipeline folder.
2. Run experiment:
`python experiment.py`