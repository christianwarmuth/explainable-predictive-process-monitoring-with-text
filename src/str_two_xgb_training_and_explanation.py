from time import time

import boto3
import shap
import sklearn
import torch
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from src.preparation_pipeline import PreparationPipeline
from src.utils import *
from src.utils import plot_confusion_matrix

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import pandas as pd
from sagemaker import get_execution_role

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


class XGB_Training_and_Explanation:

    @staticmethod
    def build(strategy: str) -> None:
        """
        Train and Explain XGBoost Model for Binary Classification
        (second-stage in a two-stage architecture)
        :param strategy: dataset used for prediction
        :return: None
        """

        print("Start Strategy 2 XGB - Training and Explanation for: " + strategies[strategy])

        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        role = get_execution_role()
        data_key = 'datasets/bpic17plus_{}.csv'.format(strategies[strategy])
        data_location = 's3://{}/{}'.format(bucket, data_key)

        n = 300

        s3 = boto3.resource('s3')
        test_case_ids = pickle.loads(s3.Bucket(bucket).Object(
            "explainability/results/strategy 1/1_bert_test_case_ids_{}.pckl".format(strategies[strategy])).get()[
                                         'Body'].read())
        train_case_ids = pickle.loads(s3.Bucket(bucket).Object(
            "explainability/results/strategy 1/1_bert_train_case_ids_{}.pckl".format(strategies[strategy])).get()[
                                          'Body'].read())
        shap_values_bert = pickle.loads(s3.Bucket(bucket).Object(
            "explainability/results/strategy 1/1_bert_shap_values_{}.pckl".format(strategies[strategy])).get()[
                                            'Body'].read())

        feature_importances = get_n_important_features(shap_values_bert, n=None, with_importance=True)

        df = pd.DataFrame(feature_importances, ["feature_name", "importance"]).T
        df = df.sort_values(by='importance', ascending=False)
        df["importance"] = pd.to_numeric(df["importance"])

        df = df.loc[df['importance'] > 0.05]
        important_words = df["feature_name"].tolist()
        if len(important_words) > n:
            print("Len imp words: " + str(len(important_words)))

            important_words = important_words[0:n]

        case_id = train_case_ids + test_case_ids

        ################################################################################################################
        # Dataset Preparation
        ################################################################################################################

        # Dataset Preparation Textual Data
        X_text = PreparationPipeline.build(data_location, option="bert_language_model")
        accepted_map = X_text.groupby('Case ID').tail(1).reset_index()[["Case ID", "label"]]
        df_new = X_text.groupby('Case ID').head(1).reset_index()[["Case ID", "desc"]].merge(accepted_map, on="Case ID")
        df_new["desc"] = df_new["desc"].map(prep)

        df_new = df_new[["desc", "label", "Case ID"]].rename(columns={"desc": "text"})

        train_text, test_text = train_test_split(df_new, test_size=0.2, shuffle=True, random_state=0)

        train_labels = train_text["Case ID"].tolist()
        test_labels = test_text["Case ID"].tolist()

        if important_words:
            tfidf_vectorizer = TfidfVectorizer(vocabulary=important_words)
            X_train_text = tfidf_vectorizer.fit_transform(train_text["text"])
            X_test_text = tfidf_vectorizer.transform(test_text["text"])
            feature_names_text = tfidf_vectorizer.vocabulary_

        # Dataset Prepratation Eventlog
        X, y, case_id_map, feature_names = PreparationPipeline.build(data_location, option="static_only",
                                                                     sorting=case_id)

        if important_words:
            feature_names = feature_names + list(feature_names_text.keys())

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

        if important_words:
            X_train_n = np.concatenate((X_train, X_train_text.todense()), axis=1)
            X_test_n = np.concatenate((X_test, X_test_text.todense()), axis=1)
        else:
            X_train_n = X_train
            X_test_n = X_test

        class_weights = sklearn.utils.class_weight.compute_class_weight('balanced', np.unique(y), y)

        ################################################################################################################
        # Model Definition
        ################################################################################################################

        xgb_model = xgb.XGBClassifier(learning_rate=0.05, n_estimators=600, max_depth=10, min_child_weight=3,
                                      subsample=0.7,
                                      colsample_bytree=0.7,
                                      objective='binary:logistic',
                                      sample_weight=class_weights,
                                      verbosity=0)

        ################################################################################################################
        # Model Training
        ################################################################################################################

        start_time = time()
        xgb_model.fit(X_train_n, y_train)
        end_time = time()
        training_time = end_time - start_time

        y_pred = xgb_model.predict(X_test_n)
        score_xgb = accuracy_score(y_test, y_pred)
        print('XGBoost Classifier:       {}'.format(score_xgb))

        pickl_save("results/strategy 2/2_xgb_class_proba_{}.pckl".format(strategies[strategy]),
                   xgb_model.predict_proba(X_test_n))
        pickl_save("results/strategy 2/2_xgb_y_test_{}.pckl".format(strategies[strategy]), y_test)

        pickl_save("results/strategy 2/2_xgb_class_proba_{}.pckl".format(strategies[strategy]),
                   xgb_model.predict_proba(X_test_n))
        pickl_save("results/strategy 2/2_xgb_y_test_{}.pckl".format(strategies[strategy]), y_test)

        print(confusion_matrix(y_test, y_pred))
        plot_confusion_matrix(confusion_matrix(y_test, y_pred))

        pickl_save("results/strategy 2/2_xgb_confusion_matrix_{}.pckl".format(strategies[strategy]),
                   confusion_matrix(y_test, y_pred))

        ################################################################################################################
        # Model Explanation using SHAP
        ################################################################################################################

        explainer = shap.Explainer(xgb_model, feature_names=feature_names)
        print("Explainer")
        print(explainer)

        start_time = time()
        shap_values = explainer(X_test_n[:num_explanations], check_additivity=False)
        end_time = time()
        explain_time = end_time - start_time

        shap_values = explainer(X_test_n, check_additivity=False)
        expected_value = explainer.expected_value

        ################################################################################################################
        # Saving Results
        ################################################################################################################

        pickl_save("results/strategy 2/2_xgb_shap_expected_value_{}.pckl".format(strategies[strategy]), expected_value)

        pickl_save("results/strategy 2/2_xgb_time_{}.pckl".format(strategies[strategy]),
                   [strategies[strategy], num_training_samples, training_time, num_explanations, explain_time])
        pickl_save("results/strategy 2/2_xgb_shap_values_{}.pckl".format(strategies[strategy]), shap_values)
        pickl_save("results/strategy 2/2_xgb_feature_importances_{}.pckl".format(strategies[strategy]),
                   xgb_model.feature_importances_)
        pickl_save("results/strategy 2/2_xgb_feature_names_{}.pckl".format(strategies[strategy]), feature_names)

        upload_to_s3('explainability', "results/strategy 2/2_xgb_confusion_matrix_{}.pckl".format(strategies[strategy]))
        upload_to_s3('explainability', "results/strategy 2/2_xgb_time_{}.pckl".format(strategies[strategy]))
        upload_to_s3('explainability', "results/strategy 2/2_xgb_shap_values_{}.pckl".format(strategies[strategy]))
        upload_to_s3('explainability', "results/strategy 2/2_xgb_class_proba_{}.pckl".format(strategies[strategy]))
        upload_to_s3('explainability', "results/strategy 2/2_xgb_y_test_{}.pckl".format(strategies[strategy]))
        upload_to_s3('explainability',
                     "results/strategy 2/2_xgb_feature_importances_{}.pckl".format(strategies[strategy]))
        upload_to_s3('explainability', "results/strategy 2/2_xgb_feature_names_{}.pckl".format(strategies[strategy]))

        upload_to_s3('explainability',
                     "results/strategy 2/2_xgb_shap_expected_value_{}.pckl".format(strategies[strategy]))

        print("Finished Strategy 2 XGB - Training and Explanation for: " + strategies[strategy])
