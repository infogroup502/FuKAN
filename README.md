# FuKAN
This repository contains the official implementation for the paper [Fuzzy KAN-Based Bidirectional Fast Anomaly Detection for Multi-Sensor Signals]().

## Requirements
The recommended requirements for FuKAN are specified as follows:
- arch==6.1.0
- einops==0.6.1
- matplotlib==3.7.0
- numpy==1.23.5
- pandas==1.5.3
- Pillow==9.4.0
- scikit_learn==1.2.2
- scipy==1.8.1
- statsmodels==0.14.0
- torch==1.13.0
- tqdm==4.65.0
- tsfresh==0.20.1


The dependencies can be installed by:
```bash
pip install -r requirements.txt
```
## Data 
The datasets can be obtained and put into datasets/ folder in the following way:
- For univariate datasets : You can download at (https://github.com/TheDatumOrg/TSB-UAD) and split them  80% into training set (_<datasaet>_train.npy) and 20% into test set (_<datasaet>_test.npy), and save the labels out as (< datasaet>_test_label.npy)
- For multivariate datasets : - [SKAB](https://github.com/waico/SkAB) should be placed at `datasets/SKAB/`.
                              - [PUMP](https://www.kaggle.com/datasets/nphantawee/pump-sensor-data) should be placed at `datasets/PUMP/`.
                              - [WaDi](https://itrust.sutd.edu.sg/itrust-labs_datasets/) should be placed at `datasets/WaDi/`.
                              - [SWAT](https://drive.google.com/drive/folders/1ABZKdclka3e2NXBSxS9z2YF59p7g2Y5I) should be placed at `datasets/SWAT/`.


## Code Description
There are six files/folders in the source
- data_factory: The preprocessing folder/file. All datasets preprocessing codes are here.
- main.py: The main python file. You can adjustment all parameters in there.
- metrics: There is the evaluation metrics code folder.
- model: FuKAN model folder
- solver.py: Another python file. The training, validation, and testing processing are all in there
- requirements.txt: Python packages needed to run this repo
## Usage
1. Install Python 3.6, PyTorch >= 1.4.0
2. Download the datasets
3. To train and evaluate FuKAN on a dataset, run the following command:
```bash
python main.py 
```
