from warnings import simplefilter

simplefilter(action='ignore', category=DeprecationWarning)

from src.preparation_pipeline import PreparationPipeline

from src.utils import plot_confusion_matrix
import sklearn
from sklearn.utils.class_weight import compute_class_weight

from sagemaker import get_execution_role

import boto3
import pickle
import numpy as np
import xgboost as xgb
from time import time
import shap
import torch

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

strategies = ["impure_0.0", "impure_0.1", "impure_0.2", "impure_0.3", "impure_0.4", "impure_0.5", "impure_0.6",
              "impure_0.7", "impure_0.8", "impure_0.9", "impure_1.0"]
num_training_samples = 5000
num_eval_samples = 5000
num_explanations = 500

bucket = 'BUCKET_NAME'  # TODO Change to Bucket Name


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


class XGB_Training_and_Explanation:

    @staticmethod
    def build(strategy: str) -> None:
        """
        Train and Explain XGBoost Model for Binary Classification
        :param strategy: dataset used for prediction
        :return: None
        """

        print("Start Strategy 1 XGB - Training and Explanation for: " + strategies[strategy])

        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        role = get_execution_role()
        data_key = 'datasets/bpic17plus_{}.csv'.format(strategies[strategy])
        data_location = 's3://{}/{}'.format(bucket, data_key)

        s3 = boto3.resource('s3')
        test_case_ids = pickle.loads(s3.Bucket(bucket).Object(
            "explainability/results/strategy 1/1_bert_test_case_ids_{}.pckl".format(strategies[strategy])).get()[
                                         'Body'].read())
        train_case_ids = pickle.loads(s3.Bucket(bucket).Object(
            "explainability/results/strategy 1/1_bert_train_case_ids_{}.pckl".format(strategies[strategy])).get()[
                                          'Body'].read())

        case_id = train_case_ids + test_case_ids

        ################################################################################################################
        # Dataset Preparation
        ################################################################################################################

        X, y, case_id_map, feature_names = PreparationPipeline.build(data_location, option="static_only",
                                                                     sorting=case_id)
        X[:, 145:166] = X[:, 145:166].astype(int)

        case_id_map["train"] = case_id_map['Case ID'].isin(train_case_ids)
        condition_test = case_id_map["train"] == True
        condition_train = case_id_map["train"] == False

        delete_test = case_id_map.index[condition_test]
        delete_train = case_id_map.index[condition_train]

        delete_list_train = delete_train.tolist()
        delete_list_test = delete_test.tolist()

        X_test = np.delete(X, delete_list_test, 0)
        y_test = np.delete(y, delete_list_test, 0)

        X_train = np.delete(X, delete_list_train, 0)
        y_train = np.delete(y, delete_list_train, 0)

        class_weights = sklearn.utils.class_weight.compute_class_weight('balanced', np.unique(y), y)

        ################################################################################################################
        # Model Definition
        ################################################################################################################

        xgb_model = xgb.XGBClassifier(learning_rate=0.05, n_estimators=500, max_depth=10, min_child_weight=5,
                                      subsample=0.8,
                                      colsample_bytree=0.8,
                                      objective='binary:logistic',
                                      sample_weight=class_weights,
                                      verbosity=0)

        ################################################################################################################
        # Model Training
        ################################################################################################################

        start_time = time()
        xgb_model.fit(X_train, y_train)
        end_time = time()
        training_time = end_time - start_time

        from sklearn.metrics import accuracy_score
        y_pred = xgb_model.predict(X_test)
        score_xgb = accuracy_score(y_test, y_pred)
        print('XGBoost Classifier:       {}'.format(score_xgb))

        pickl_save("results/strategy 1/1_xgb_class_proba_{}.pckl".format(strategies[strategy]),
                   xgb_model.predict_proba(X_test))
        pickl_save("results/strategy 1/1_xgb_y_test_{}.pckl".format(strategies[strategy]), y_test)

        from sklearn.metrics import confusion_matrix
        plot_confusion_matrix(confusion_matrix(y_test, y_pred))
        pickl_save("results/strategy 1/1_xgb_confusion_matrix_{}.pckl".format(strategies[strategy]),
                   confusion_matrix(y_test, y_pred))

        ################################################################################################################
        # Model Explanation using SHAP
        ################################################################################################################

        explainer = shap.Explainer(xgb_model, feature_names=feature_names)

        start_time = time()
        shap_values = explainer(X_test[:num_explanations], check_additivity=False)
        end_time = time()
        explain_time = end_time - start_time

        shap_values = explainer(X_test, check_additivity=False)

        ################################################################################################################
        # Saving Results
        ################################################################################################################

        pickl_save("results/strategy 1/1_xgb_time_{}.pckl".format(strategies[strategy]),
                   [strategies[strategy], num_training_samples, training_time, num_explanations, explain_time])

        pickl_save("results/strategy 1/1_xgb_shap_values_{}.pckl".format(strategies[strategy]), shap_values)

        pickl_save("results/strategy 1/1_xgb_feature_importances_{}.pckl".format(strategies[strategy]),
                   xgb_model.feature_importances_)
        pickl_save("results/strategy 1/1_xgb_feature_names_{}.pckl".format(strategies[strategy]), feature_names)

        upload_to_s3('explainability', "results/strategy 1/1_xgb_confusion_matrix_{}.pckl".format(strategies[strategy]))
        upload_to_s3('explainability', "results/strategy 1/1_xgb_time_{}.pckl".format(strategies[strategy]))
        upload_to_s3('explainability', "results/strategy 1/1_xgb_shap_values_{}.pckl".format(strategies[strategy]))
        upload_to_s3('explainability', "results/strategy 1/1_xgb_class_proba_{}.pckl".format(strategies[strategy]))
        upload_to_s3('explainability', "results/strategy 1/1_xgb_y_test_{}.pckl".format(strategies[strategy]))

        upload_to_s3('explainability',
                     "results/strategy 1/1_xgb_feature_importances_{}.pckl".format(strategies[strategy]))
        upload_to_s3('explainability', "results/strategy 1/1_xgb_feature_names_{}.pckl".format(strategies[strategy]))

        print("Finished Strategy 1 XGB - Training and Explanation for: " + strategies[strategy])
