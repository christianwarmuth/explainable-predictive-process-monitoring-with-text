from typing import List

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from src.utils import preprocessing_pipeline, clean_desc


class PreparationPipeline:
    @staticmethod
    def build(event_log_path: str, option: str = "static_only", sorting: List[str] = None,
              random_seed: int = 1):
        """
        Build preprocessing Pipeline & perform the specified preprocessing.
        :param sorting: consistent sorting for two models/two-stage models later-on
        :param event_log_path: Path to the extended eventlog
        :param option: Option to differentiate between the different possible preprocessing steps.
        Choices: "bert_language_mobel", "normal_language_model", "static_only", "sequence_only"
        :param random_seed: Seed value for random methods & parameters
        :return: X, y, case_id_map, feature_names, **
        """

        options = ["bert_language_model", "normal_language_model", "sequence_only", "static_only"]

        if option == "bert_language_model":
            # Returning data without preprocessing (one column text, one column labels)
            X = PreparationPipeline.load_and_preprocess_extended_eventlog_dataset(
                event_log_path, option)
            return X

        elif option == "normal_language_model":
            # Returning tf-idf-matrix & labels
            X, y, case_id_map, feature_names, corpus_desc_str = PreparationPipeline.load_and_preprocess_extended_eventlog_dataset(
                event_log_path, option)
            return X, y, case_id_map, feature_names, corpus_desc_str

        elif option == "static_only":
            # Returning only eventlog with static encoding technique
            X, y, case_id_map, feature_names = PreparationPipeline.load_and_preprocess_extended_eventlog_dataset(
                event_log_path, option, sorting
            )
            return X, y, case_id_map, feature_names

        elif option == "sequence_only":
            # Returning only eventlog with sequence encoding technique
            df = pd.read_csv(event_log_path, delimiter=";")
            # df = PreparationPipeline.load_and_preprocess_extended_eventlog_dataset(event_log_path, option)
            # TODO: Implement sequence preprocessing
            return df
        else:
            print("No valid option chosen")
            return

    @staticmethod
    def load_and_preprocess_extended_eventlog_dataset(event_log_path: str,
                                                      option: str = "static_only", sorting: List[str] = None,
                                                      ) -> pd.DataFrame:
        """
        Detailed Preprocessing steps.
        :param sorting: consistent sorting for two models/two-stage models later-on
        :param event_log_path: Path to the extended eventlog
        :param option: Option to differentiate between the different possible preprocessing steps.
         Choices: "bert_language_mobel", "normal_language_model", "static_only", "sequence_only"
        :return: X, y, case_id_map, feature_names, **
        """

        df = pd.read_csv(event_log_path, delimiter=";")
        columns_min = ["Case ID", "event_nr", "Activity"]
        columns_cat = ["label", "LoanGoal", "ApplicationType",
                       "weekday", "Action", "org:resource", "EventOrigin", "lifecycle:transition"]
        columns_num = ["RequestedAmount", "FirstWithdrawalAmount", "MonthlyCost", "NumberOfTerms", "OfferedAmount",
                       "CreditScore", "open_cases", "timesincelastevent", "timesincecasestart", "timesincemidnight"]
        columns_txt = ["desc", "title", "emp_title"]
        column_list = columns_min + columns_cat + columns_num + columns_txt

        # Drop unwanted columns
        df = df[column_list]

        if option == "bert_language_model":
            df["desc"] = df["desc"].map(clean_desc)
            return df[["Case ID", "Activity", "desc", "label"]]

        if option == "static_only":
            X, y, case_id_map, feature_names = PreparationPipeline.preprocess_event_log(df,
                                                                                        sorting=sorting)
            return X, y, case_id_map, feature_names


        elif option == "normal_language_model":

            X, y, case_id_map, feature_names, corpus_desc_str = PreparationPipeline.preprocess_text_data(df)

            return X, y, case_id_map, feature_names, corpus_desc_str

        else:
            return

    @staticmethod
    def preprocess_event_log(df: pd.DataFrame, sorting: List[str] = None):
        """
        Non-Textual data preprocessing
        :param sorting: consistent sorting for two models/two-stage models later-on
        :param df: Dataframe as input
        :return:  X, y, case_id_map, feature_names
        """

        df = df.drop(["desc", "title", "emp_title"], axis=1)
        case_id_map = df.groupby('Case ID').tail(1).reset_index()[["Case ID", "label"]]

        cutoff_events = ["O_Accepted", "A_Cancelled", "A_Denied"]

        def row_cutof(grouped_df):
            idx = grouped_df.reset_index().index[grouped_df["Activity"].isin(cutoff_events)]

            if len(idx) == 0:
                # print(grouped_df)
                print("No cutoff event found ->  ignore case")
                return grouped_df
            return grouped_df.iloc[:idx[0]]

        df = df.groupby(["Case ID"]).apply(row_cutof).reset_index(drop=True)

        variables_to_oh = ["Activity", "LoanGoal", "ApplicationType", "weekday", "Action", "org:resource",
                           "EventOrigin", "lifecycle:transition"]
        variables_to_scale = ["RequestedAmount", "FirstWithdrawalAmount", "MonthlyCost", "NumberOfTerms",
                              "OfferedAmount",
                              "CreditScore", "open_cases", "timesincelastevent", "timesincecasestart",
                              "timesincemidnight"]

        aggregation_types = {
            'LoanGoal': 'first',
            'RequestedAmount': 'first',
            'open_cases': 'last',
            'timesincelastevent': 'last',
            'timesincecasestart': 'last',
            'timesincemidnight': 'first',
            'FirstWithdrawalAmount': 'first',
            'MonthlyCost': 'first',
            'NumberOfTerms': 'first',
            'OfferedAmount': 'first',
            'CreditScore': 'first',
            'ApplicationType': 'first',
            'weekday': 'first',
            'Action': "first",
            "org:resource": "first",
            "EventOrigin": "first"
        }

        df = df.drop(["event_nr"], axis=1)
        traces = df.groupby("Case ID", as_index=False).agg(aggregation_types)

        activity_types = df["Activity"].unique()
        activities = df.groupby("Case ID").agg({"Activity": lambda x: list(x)})

        activities = pd.DataFrame(activities.reset_index())

        for activity in activity_types:
            activities[activity] = activities["Case ID"].apply(lambda x: activity in x)
        activities.drop('Activity', axis=1, inplace=True)

        # Scaling variables
        scaler = StandardScaler()
        traces[variables_to_scale] = scaler.fit_transform(traces[variables_to_scale])
        traces = pd.get_dummies(traces, columns=["LoanGoal", "ApplicationType", "weekday", "Action", "org:resource",
                                                 "EventOrigin"])
        traces = traces.merge(activities, on='Case ID')

        event_log_pre = traces.merge(case_id_map, on='Case ID')

        if sorting is not None:
            event_log_pre.columns = event_log_pre.columns.to_series().apply(lambda x: x.strip())
            sorterIndex = dict(zip(sorting, range(len(sorting))))
            event_log_pre['ID_rank'] = event_log_pre['Case ID'].map(sorterIndex)
            event_log_pre.sort_values(['ID_rank'], inplace=True)
            event_log_pre.drop('ID_rank', 1, inplace=True)
            event_log_pre = event_log_pre.reset_index()

        # Prepare eventlog:
        y = event_log_pre["label"].to_numpy().astype('float32')
        case_id_map = event_log_pre[["Case ID", "label"]]
        event_log_pre.drop('index', 1, inplace=True)

        event_log_pre.drop(["Case ID", "label"], axis=1, inplace=True)
        feature_names = event_log_pre.columns.to_list()

        X = event_log_pre.to_numpy()
        return X, y, case_id_map, feature_names

    @staticmethod
    def preprocess_text_data(df: pd.DataFrame):
        """
        Textual Data Preprocessing
        :param df: Dataframe as input
        :return: X, y, case_id_map, feature_names, corpus_desc_str
        """
        case_id_map = df.groupby('Case ID').tail(1).reset_index()[["Case ID", "label"]]

        df = df.groupby('Case ID').head(1).reset_index()  # Textual Data on case-level (first event per case)
        df = df[["desc", "title", "emp_title", "Case ID"]]
        df = df.merge(case_id_map, on='Case ID')
        df["desc"] = df["desc"].map(clean_desc)

        corpus_desc = []
        corpus_title = []
        corpus_emp_title = []

        for element in df["desc"].tolist():
            corpus_desc.append(element.split())

        for element in df["title"].tolist():
            corpus_title.append(element.split())

        for element in df["emp_title"].tolist():
            corpus_emp_title.append(str(element).split())

        corpus_desc = preprocessing_pipeline(corpus_desc)
        corpus_title = preprocessing_pipeline(corpus_title)
        corpus_emp_title = preprocessing_pipeline(corpus_emp_title)
        y = df["label"].to_numpy().astype('float32')
        corpus_desc_str = []
        for element in corpus_desc:
            corpus_desc_str.append(' '.join(element))

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(corpus_desc_str)
        feature_names = vectorizer.get_feature_names()

        return X, y, case_id_map, feature_names, corpus_desc_str
