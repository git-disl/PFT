
# Matching Accounts on Blockchain with Pseudo Fine-tuning of Language Models

Code and dataset for the submission "Matching Accounts on Blockchain with Pseudo Fine-tuning of Language Models".

## Getting Start
### Requirements:
* Python >= 3.6.1
* NumPy >= 1.18.1
* TensorFlow >= 1.4.0

###  1. Preprocess dataset 

#### Step 1. Download dataset from Google drive:
* [All_in_one](https://drive.google.com/file/d/1LPloeakxyp00Ez56EnjpuSQ6LCTLWUMz/view?usp=share_link)
* [Typhoon dataset (new)](https://drive.google.com/file/d/1WzCXxPGO1dDyfMZ7CqAKvgfiSVM2lyoZ/view?usp=sharing)

#### Step 2. Unzip dataset under the directory of "PFT/Data"
``` 
tar -xvf PFT_Data.tar.gz
``` 

### 2. Pseudo-supervised learning

Please refer to ./Model/run_pplm.sh