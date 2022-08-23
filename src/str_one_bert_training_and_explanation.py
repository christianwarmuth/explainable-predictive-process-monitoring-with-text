import pickle
from time import time

import boto3
import gensim
import numpy as np
import pandas as pd
import scipy as sp
import shap
import torch
import torch.nn as nn
from datasets import Dataset
from datasets import Features, Value
from datasets import load_metric
from sagemaker import get_execution_role
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoConfig
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import Trainer
from transformers import TrainingArguments

from src.preparation_pipeline import PreparationPipeline

strategies = ["impure_0.0", "impure_0.1", "impure_0.2", "impure_0.3", "impure_0.4", "impure_0.5", "impure_0.6",
              "impure_0.7", "impure_0.8", "impure_0.9", "impure_1.0"]
num_training_samples = 5000
num_eval_samples = 5000
num_explanations = 500

bucket = 'BUCKET_NAME'  # TODO Change to Bucket Name


def prep(desc: str) -> str:
    """
    Simple Preprocessing using Gemsim
    :param desc: input text
    :return: preprocessed text
    """

    if pd.isna(desc):
        return desc
    else:
        return ' '.join(gensim.utils.simple_preprocess(desc))


def pickl_save(location: str, obj) -> None:
    """
    Save Object in .pckl format
    :param location: Save Location
    :param obj: Object to be saved
    :return: None
    """

    pickl_location = location
    with open(pickl_location, "wb") as f:
        pickle.dump(obj, f)


def upload_to_s3(channel: str, file: str) -> None:
    """
    Upload file from AWS Sagemaker to S3
    :param channel: Save folder
    :param file: Source folder
    :return: None
    """
    s3 = boto3.resource('s3')
    data = open(file, "rb")
    key = channel + '/' + file
    s3.Bucket(bucket).put_object(Key=key, Body=data)


class Bert_Training_and_Explanation:

    @staticmethod
    def build(strategy: str) -> None:
        """
        Train and Explain BERT Model for Binary Classification
        :param strategy: dataset used for prediction
        :return: None
        """

        print("Start Strategy 1 BERT - Training and Explanation for: " + strategies[strategy])

        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        role = get_execution_role()
        data_key = 'datasets/bpic17plus_{}.csv'.format(strategies[strategy])
        data_location = 's3://{}/{}'.format(bucket, data_key)

        ################################################################################################################
        # Dataset Preparation
        ################################################################################################################

        X = PreparationPipeline.build(data_location, option="bert_language_model")
        accepted_map = X.groupby('Case ID').tail(1).reset_index()[["Case ID", "label"]]
        df_new = X.groupby('Case ID').head(1).reset_index()[["Case ID", "desc"]].merge(accepted_map, on="Case ID")
        df_new["desc"] = df_new["desc"].map(prep)

        df_new = df_new[["desc", "label", "Case ID"]].rename(columns={"desc": "text"})

        train, test = train_test_split(df_new, test_size=0.2, shuffle=True, random_state=0)

        train_labels = train["Case ID"].tolist()
        test_labels = test["Case ID"].tolist()

        pickl_save("results/strategy 1/1_bert_train_case_ids_{}.pckl".format(strategies[strategy]), train_labels)
        pickl_save("results/strategy 1/1_bert_test_case_ids_{}.pckl".format(strategies[strategy]), test_labels)

        upload_to_s3('explainability', "results/strategy 1/1_bert_train_case_ids_{}.pckl".format(strategies[strategy]))
        upload_to_s3('explainability', "results/strategy 1/1_bert_test_case_ids_{}.pckl".format(strategies[strategy]))

        dataset_train = Dataset.from_pandas(train, features=Features({'text': Value(dtype='string', id=None),
                                                                      'label': Value(dtype='int32', id=None)}),
                                            split="train")
        dataset_test = Dataset.from_pandas(test, features=Features({'text': Value(dtype='string', id=None),
                                                                    'label': Value(dtype='int32', id=None)}),
                                           split="test")

        train, test = train_test_split(df_new, test_size=0.2, shuffle=True, random_state=0)

        train_labels = train["Case ID"].tolist()
        test_labels = test["Case ID"].tolist()

        pickl_save("results/strategy 1/1_bert_train_case_ids_{}.pckl".format(strategies[strategy]), train_labels)
        pickl_save("results/strategy 1/1_bert_test_case_ids_{}.pckl".format(strategies[strategy]), test_labels)

        upload_to_s3('explainability', "results/strategy 1/1_bert_train_case_ids_{}.pckl".format(strategies[strategy]))
        upload_to_s3('explainability', "results/strategy 1/1_bert_test_case_ids_{}.pckl".format(strategies[strategy]))

        dataset_train = Dataset.from_pandas(train, features=Features({'text': Value(dtype='string', id=None),
                                                                      'label': Value(dtype='int32', id=None)}),
                                            split="train")
        dataset_test = Dataset.from_pandas(test, features=Features({'text': Value(dtype='string', id=None),
                                                                    'label': Value(dtype='int32', id=None)}),
                                           split="test")

        ################################################################################################################
        # Model Definition
        ################################################################################################################

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        def tokenize_function(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True)

        tokenized_datasets_train = dataset_train.map(tokenize_function, batched=True)
        tokenized_datasets_test = dataset_test.map(tokenize_function, batched=True)

        small_train_dataset = tokenized_datasets_train.shuffle(seed=42).select(range(num_training_samples))
        small_eval_dataset = tokenized_datasets_test.shuffle(seed=42).select(range(num_eval_samples))
        full_train_dataset = tokenized_datasets_train
        full_eval_dataset = tokenized_datasets_test

        model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased").to(device)

        config = AutoConfig.from_pretrained("bert-base-uncased")

        print("Config")
        print(config)

        training_args = TrainingArguments("test_trainer")

        # Compute Class weights
        class_wts = compute_class_weight('balanced', np.unique(dataset_train["label"]), dataset_train["label"])
        weights = torch.tensor(class_wts, dtype=torch.float)
        # Weighted Cross Entropy Loss
        cross_entropy = nn.CrossEntropyLoss(weight=weights)

        class BertTrainer(Trainer):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def compute_loss(self, model, inputs, return_outputs=False):
                labels = inputs.get("labels").to('cpu')
                outputs = model(**inputs)
                logits = outputs.get('logits').to('cpu')
                loss = cross_entropy(logits, labels)
                return (loss, outputs) if return_outputs else loss

        ################################################################################################################
        # Model Training
        ################################################################################################################

        trainer = BertTrainer(
            model=model, args=training_args, train_dataset=small_train_dataset, eval_dataset=small_eval_dataset
        )

        start_time = time()
        trainer.train()
        end_time = time()
        training_time = end_time - start_time

        trainer.save_model("1_bert_model_{}".format(strategies[strategy]))

        metric = load_metric("accuracy")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            pickl_save("results/strategy 1/1_bert_logits_{}.pckl".format(strategies[strategy]), logits)
            pickl_save("results/strategy 1/1_bert_confusion_matrix_{}.pckl".format(strategies[strategy]),
                       confusion_matrix(labels, predictions))
            return metric.compute(predictions=predictions, references=labels)

        trainer = BertTrainer(
            model=model,
            args=training_args,
            train_dataset=small_train_dataset,
            eval_dataset=full_eval_dataset,
            compute_metrics=compute_metrics,
        )
        trainer.evaluate()

        ################################################################################################################
        # Model Explanation using SHAP
        ################################################################################################################

        def f(x):
            tv = torch.tensor([tokenizer.encode(v, padding='max_length', truncation=True) for v in x]).to(device)
            outputs = model(tv)[0].detach().cpu().numpy()
            scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
            val = sp.special.logit(scores[:, 1])
            return val

        explainer = shap.Explainer(f, tokenizer)
        print("Explainer")
        print(explainer)

        start_time = time()
        shap_values = explainer(dataset_test[:num_explanations])
        end_time = time()
        explain_time = end_time - start_time

        ################################################################################################################
        # Saving Results
        ################################################################################################################

        pickl_save("results/strategy 1/1_bert_shap_values_{}.pckl".format(strategies[strategy]), shap_values)
        pickl_save("results/strategy 1/1_bert_shap_values_case_ids_{}.pckl".format(strategies[strategy]),
                   test.head(num_explanations)["Case ID"].tolist())

        pickl_save("results/strategy 1/1_bert_dataset_for_expl_{}.pckl".format(strategies[strategy]),
                   dataset_test[:num_explanations])

        pickl_save("results/strategy 1/1_bert_time_{}.pckl".format(strategies[strategy]),
                   [strategies[strategy], num_training_samples, training_time, num_explanations, explain_time])

        upload_to_s3('explainability',
                     "results/strategy 1/1_bert_shap_values_case_ids_{}.pckl".format(strategies[strategy]))
        upload_to_s3('explainability',
                     "results/strategy 1/1_bert_dataset_for_expl_{}.pckl".format(strategies[strategy]))
        upload_to_s3('explainability', "results/strategy 1/1_bert_shap_values_{}.pckl".format(strategies[strategy]))
        upload_to_s3('explainability',
                     "results/strategy 1/1_bert_confusion_matrix_{}.pckl".format(strategies[strategy]))

        upload_to_s3('explainability', "results/strategy 1/1_bert_logits_{}.pckl".format(strategies[strategy]))
        upload_to_s3('explainability', "results/strategy 1/1_bert_time_{}.pckl".format(strategies[strategy]))

        print("Finished Strategy 1 BERT - Training and Explanation for: " + strategies[strategy])
