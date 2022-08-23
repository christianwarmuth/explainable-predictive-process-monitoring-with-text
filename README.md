# On the Potential of Textual Data for Explainable Predictive Process Monitoring
This is the supplementary repository for the research "On the Potential of Textual Data for Explainable Predictive Process Monitoring". Submitted to [ICPM Conference 2022](https://icpmconference.org/2022/) (3rd International [Workshop on Leveraging Machine Learning in Process Mining (ML4PM)](http://ml4pm2022.di.unimi.it/)).



- Author: Christian Warmuth, Prof. Dr. Henrik Leopold
- Affiliation: Hasso Plattner Institute
- Chair: Business Process Technology


## Abstract: 

Predictive process monitoring techniques leverage machine learning (ML) to predict future characteristics of a case, such as the process outcome or the remaining run time. Available techniques employ various models and different types of input data to produce accurate predictions. However, from a practical perspective, explainability is another important requirement besides accuracy since predictive process monitoring techniques frequently support decision-making in critical domains. 

Techniques from the area of explainable artificial intelligence (XAI) aim to provide this capability and create transparency and interpretability for black-box ML models. While several explainable predictive process monitoring techniques exist, none of them leverages textual data. This is surprising since textual data can provide a rich context to a process that numerical features cannot capture. 

Recognizing this, we use this paper to investigate how the combination of textual and non-textual data can be used for explainable predictive process monitoring and analyze how the incorporation of textual data affects both the predictions and the explainability. Our experiments show that using textual data requires more computation time but can lead to a notable improvement in prediction quality with comparable results for explainability. 


## Setup
The dataset augmentation can be run a local device, while the experiments were conducted on AWS Sagemaker with a g4dn.2xlarge Instance.

```shell
conda create -n xai python=3.7
conda activate xai
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

To use Topic Modeling using LDA (and optimized LDA with gibbs sampling from MALLET)
```shell
conda install -c conda-forge pyldavis
```
```
Download from "http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip" and set in path in config.toml
```

For Experiment Evaluation:
```
Download pretrained word2vec vectors from https://code.google.com/archive/p/word2vec/ and set in path in config.toml
```

## Experimental Setup

The experiments are conducted on an Amazon Webservices (AWS) Sagemaker cloud computing instance ([ml.g4dn.2xlarge](https://aws.amazon.com/de/ec2/instance-types/g4/)) with 8x Intel Xeon Scalable Processors of the 2nd Generation. The server is equipped with 1x NVIDIA T4 Tensor Core GPU with 16 GiB of high bandwidth graphic memory (GDDR6) and 32 GiB RAM. CUDA version is 11.4, and the driver version is 470.57.02 with Python 3.7.10. We use Python's integrated time package for time measurements. The calculations were performed with a random seed for all experiments to ensure reproducibility.

For the first stage of strategy 2, we use a BERT model for filtering out the $n$ most important words, which are fed into a second-stage XGBoost model. For strategy 2, we will take the first 300 most important features into the second stage XGBoost model.

For the experiments, we will perform the different evaluations on 11 synthetically augmented datasets with an impurity ranging from 0.0 to 1.0 in steps of 0.1. For simplification reasons, we use the notion of purity as *1 - impurity*.

## Repo Organization

    ├── README.md          <- Top-level README.
    ├── charts             <- Charts generated for the publication and experiments.
    ├── experiments        <- Jupyter notebooks for the experiments to be run on AWS Sagemaker.
    ├── exploration        <- Jupyter notebooks containing exploratory data analysis as well as prototype impementations. 
    ├── config.toml        <- Config file to specify paths to files.
    ├── requirements.txt   <- The requirements file for reproducing the analysis.
    └── src                <- Source code for experiments, exploratory data analysis and the prototype.


