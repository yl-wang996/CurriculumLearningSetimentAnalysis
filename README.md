



# Curriculum Learning in Sentiment Analysis

  

## 0. Declare

  
Our model is base on the [pretraining model](https://huggingface.co/distilbert-base-uncased) to fine tune on the [IMDB dataset](http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz).

  

>  **Author** : 	Yuqian Lei (yuqian.lei@studium.uni-hamburg.de)
							Yunlong Wang (yunlong.wang@studium.uni-hamburg.de)

  
  
  

## 1. Introduction

  

We implement the the curriculum learing on the fine tuning stage of the distilBERT in task sentiment analysis. we find the curriculum learning although only have marginal beneftis by finding the optimal solution, but good at stabilizing the optimization when noise is added. Our code is submit in our [git](https://git.mafiasi.de/working_team/CLinSematicClassification.git).

  

## 2. Environment

  

The file `venv.yml` contain all the required package which directly export from anaconda command line. You can also use anaconda to rebuild the virtual environment.

  

## 3. Dataset
### 3.1 Download dataset
Download data from [here](http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz), which include two file. Put them into your data folder.
### 3.2 Sub-Dataset generation
Modify the path related to data folder in `preprocessing.py`, and run to generate the dataset for usage(2500 for train and validation, 500 for test).

## 4. Run
> For each step, you need to config path first.
 1.  Firstly, run `loss-based-cscore.py` to generate the diffculty of each eaxmlpes for training.
 2. Then, run `main_w_test.py` to calculate the result, it may takes long time.
