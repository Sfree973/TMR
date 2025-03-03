# TMR
Source codes for TMR(IEEE Transactions on Knowledge and Data Engineering).

This open source code was published during the review of our article. Note that any falsification, plagiarism, and commercial use are not permitted until the article is accepted.

### Quick Start
Installation: 
Install PyTorch following the instructions on the PyTorch. Our code is written in Python3.

Run the following commands to install the required packages:
```
pip install -r requirements.txt
```

### Data
```
unzip data.zip
```
It will generate six inductive dataset folders (wnv1,wnv2,wnv3,fbv1,fbv2,fbv3) in the ./data directory. Our transductive data used in the experiments as well as the different embedding files can be found here: https://fileserver.ukp.informatik.tu-darmstadt.de/starsem18-multimodalKB/.

Furthermore, for your convenience, the training and testing commands of the code are as follows： 
### Train and test models
1. Train our model
```
python src/experiments.py --train --dataset <dataset-name> --gpu <gpu-ID>
```

2. Evaluate our model 
```
python src/experiments.py --inference --dataset <dataset-name> --gpu <gpu-ID>
```
