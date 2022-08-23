from __future__ import annotations

import random
import sys
from collections import Counter
from typing import List, Tuple

import pandas as pd
import toml

config = toml.load("../config.toml")

keyword_map_title = {"Home improvement": ["home", "bedroom", "bathroom", "basement", "kitchen", "floor",
                                          "property", "house", "relocation", "remodel",
                                          "renovation", "apartment"],
                     "Student Loan": ["student", "fee", "university", "tuition", "school", "degree", "class", "grad",
                                      "graduate"],
                     "Consume": ["mustang", "car", "machine", "auto", "purchase", "replacement", "sport", "christmas",
                                 "game", "gift", "bike", "scooter"],
                     "Medical": ["hospital", "cancer", "medical", "doctor", "uninsured",
                                 "medicine", "surgery", "insurance", "drug", "treatment", "dental"],
                     "Vacation": ["vacation", "summer", "winter", "country", "travel", "family", "wedding", "ring",
                                  "swim", "pool", "hotel"],
                     "Consolidation": ["refinance", "debt", "interest", "consolidation", "banks", "rate", "cut",
                                       "payoff", "limit", "reduction", "credit"],
                     }

split = [(1.0, 0.0), (0.0, 1.0), (1.0, 0.0), (0.0, 1.0), (1.0, 0.0), (0.0, 1.0)]


class DatasetEnrichment:
    @staticmethod
    def build(event_log_path: str, text_dataset_path: str, strategy: str, impurity: float, out_path: str) -> None:
        """
        Dataset Augmentation
        :param event_log_path: Path to the eventlog dataset
        :param text_dataset_path: Path to the LendingClub dataset
        :param strategy: matching strategy (random, static, topical)
        :param impurity: impurity of augmented dataset
        :param out_path: target path of the enriched dataset
        :return: None
        """

        print("Start enriching dataset with strategy: " + str(strategy))
        event_log = pd.read_csv(event_log_path, sep=";")
        text_dataset = pd.read_csv(text_dataset_path)

        enriched_dataset = DatasetEnrichment.dataset_enrichment(event_log, text_dataset, strategy, impurity)
        output_path = out_path + "bpic17plus_" + strategy + "_" + str(impurity) + ".csv"
        DatasetEnrichment.save_to_file(enriched_dataset, output_path)
        print("Saved dataset to path: " + output_path)

    @staticmethod
    def dataset_enrichment(event_log: pd.DataFrame, text_dataset: pd.DataFrame, strategy: str,
                           impurity: float) -> pd.DataFrame:
        """
        Implementation of the 3 strategies for augmenting eventlogs with textual data
        :param event_log: DataFrame of eventlog
        :param text_dataset: DataFrame of text dataset
        :param strategy: matching strategy (random, static, topical)
        :param impurtiy: impurity (percentage of random text)
        :return: enriched DataFrame
        """

        text_dataset['desc_word_count'] = text_dataset['desc'].str.count(' ') + 1
        text_dataset = text_dataset[
            text_dataset['desc'].notnull() & text_dataset['title'].notnull() & text_dataset['emp_title'].notnull()]
        text_dataset = text_dataset[text_dataset['desc_word_count'] > 20]

        text_dataset = text_dataset[['desc', 'title', 'emp_title']]

        enriched_dataset = ""
        if strategy not in ["random", "static", "impure"]:
            print("No valid dataset enrichment strategy")
            sys.exit()

        if strategy == "random":
            enriched_dataset = DatasetEnrichment.match_random(event_log, text_dataset)
        if strategy == "static":
            enriched_dataset = DatasetEnrichment.match_static(event_log, text_dataset, impurity)
        if strategy == "impure":
            enriched_dataset = DatasetEnrichment.match_impure(event_log, text_dataset, impurity)
        return enriched_dataset

    @staticmethod
    def match_random(event_log: pd.DataFrame, text_dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Match textual and eventlog datasets "randomly". Assign textual context data on case-level to the first activity
        per trace randomly.
        :param event_log: DataFrame of eventlog
        :param text_dataset: DataFrame of text dataset
        :return: enriched DataFrame
        """

        df = text_dataset

        # At random attach desc, title and emp_title as columns for the first event per case

        # Adding the columns to the existing dataset
        event_log["desc"] = ""
        event_log["title"] = ""
        event_log["emp_title"] = ""

        text_count = len(text_dataset)
        case_count = event_log.groupby(["Case ID"]).ngroups
        event_count = event_log.groupby(["Case ID"], sort=False).size()
        sampled_text = df.sample(n=case_count, random_state=1).values.tolist()
        sampled_text_match = []
        for idx, evt_count in enumerate(event_count):
            sampled_text_match.append(sampled_text[idx])
            for i in range(1, evt_count):
                sampled_text_match.append(["", "", ""])

        event_log = event_log.assign(text=sampled_text_match)
        split_text = pd.DataFrame(event_log["text"].tolist(), columns=["desc", "title", "emp_title"])
        event_log = pd.concat([event_log.drop(["text", "desc", "title", "emp_title"], axis=1), split_text], axis=1)
        return event_log

    @staticmethod
    def match_impure(event_log: pd.DataFrame, text_dataset: pd.DataFrame, impurity: int) -> pd.DataFrame:
        """
        Match textual and eventlog datasets similar to the topic strategy. However, instead of having a clearer class
        overlap, we introduce more impurity instead of texts relating to the other topic group . Assign textual context
        data on case-level to the first activity per trace randomly.
        :param event_log: DataFrame of eventlog
        :param text_dataset: DataFrame of text dataset
        :param impurity: impurity paramater for augmented dataset
        :return: enriched DataFrame
        """

        desc_list = text_dataset["desc"].tolist()
        event_log = event_log.replace({'label': {'deviant': 1, 'regular': 0}})
        case_id_map = event_log.groupby('Case ID').tail(1).reset_index()[["Case ID", "label"]]
        neg_case_nr = case_id_map["label"].value_counts()[0]
        pos_case_nr = case_id_map["label"].value_counts()[1]

        occurances = []
        for topic in keyword_map_title:
            occurrence_list = DatasetEnrichment.count_topic_word_occurances(desc_list, keyword_map_title[topic])
            occurrence_list.sort(key=lambda x: x[1], reverse=True)
            occurances.append(occurrence_list)

        list_to_map_accepted = []
        list_to_map_rejected = []

        # Create number of desc per topic to accepted/rejected assignment

        split = [(1.0, 0.0), (0.0, 1.0), (1.0, 0.0), (0.0, 1.0), (1.0, 0.0), (0.0, 1.0)]

        for idx, topic in enumerate(keyword_map_title):
            number_per_topic = int(split[idx][1] * (neg_case_nr / 3) + split[idx][0] * (pos_case_nr / 3))
            list_to_map_accepted.append(int(split[idx][0] * (pos_case_nr / 3)))
            list_to_map_rejected.append(number_per_topic - int(split[idx][0] * (pos_case_nr / 3)))

        text_list_acc, text_list_rej, occurances_unused = DatasetEnrichment.iterat_map_text(occurances,
                                                                                            list_to_map_accepted,
                                                                                            list_to_map_rejected,
                                                                                            return_occurances=True)
        unused_text = [item for sublist in occurances_unused for item in sublist]
        text_list_acc_flat = [item for sublist in text_list_acc for item in sublist]
        text_list_rej_flat = [item for sublist in text_list_rej for item in sublist]

        # Ensure that all cases have a text
        if len(text_list_rej_flat) > neg_case_nr:
            text_list_rej_flat = text_list_rej_flat[:neg_case_nr]
        if len(text_list_acc_flat) > pos_case_nr:
            text_list_acc_flat = text_list_acc_flat[:pos_case_nr]
        if len(text_list_acc_flat) < pos_case_nr:
            difference = pos_case_nr - len(text_list_acc_flat)
            for i in range(difference):
                text_list_acc_flat.append(occurances[0].pop())
        if len(text_list_rej_flat) < neg_case_nr:
            difference = neg_case_nr - len(text_list_rej_flat)
            for i in range(difference):
                text_list_rej_flat.append(occurances[0].pop())

        random.Random(42).shuffle(unused_text)
        random.Random(42).shuffle(text_list_acc_flat)
        random.Random(42).shuffle(text_list_rej_flat)

        # Randomly replace elements per outcome according to the impurity strength
        num_acc_rand = int(pos_case_nr * impurity)
        num_rej_rand = int(neg_case_nr * impurity)

        print("Replace randomly " + str(num_acc_rand) + " out of " + str(pos_case_nr) + " for positive cases")
        print("Replace randomly " + str(num_rej_rand) + " out of " + str(neg_case_nr) + " for negative cases")

        if not impurity == 0.0:
            print("randomly added text with impurity of " + str(impurity))
            text_list_acc_flat = text_list_acc_flat[:-num_acc_rand]
            text_list_rej_flat = text_list_rej_flat[:-num_rej_rand]
            text_list_acc_flat.extend(unused_text[:num_acc_rand])
            del unused_text[:num_acc_rand]
            text_list_rej_flat.extend(unused_text[:num_rej_rand])

        # Adding the columns to the existing dataset
        event_log["desc"] = ""
        event_log["title"] = ""
        event_log["emp_title"] = ""
        textdataset = text_dataset.reset_index()

        event_count = event_log.groupby(["Case ID"], sort=False).size()
        cases = event_log.groupby(["Case ID"], sort=False).last()["label"].tolist()
        sampled_text_match = []
        for idx, evt_count in enumerate(event_count):
            if cases[idx] == 1:
                indx = text_list_acc_flat.pop()[0]
                sampled_text_match.append([textdataset.iloc[indx]["desc"], textdataset.iloc[indx]["title"],
                                           textdataset.iloc[indx]["emp_title"]])
            else:
                indx = text_list_rej_flat.pop()[0]
                sampled_text_match.append([textdataset.iloc[indx]["desc"], textdataset.iloc[indx]["title"],
                                           textdataset.iloc[indx]["emp_title"]])
            for i in range(1, evt_count):
                sampled_text_match.append(["", "", ""])

        event_log = event_log.assign(text=sampled_text_match)
        split_text = pd.DataFrame(event_log["text"].tolist(), columns=["desc", "title", "emp_title"])
        event_log = pd.concat([event_log.drop(["text", "desc", "title", "emp_title"], axis=1), split_text], axis=1)
        return event_log

    @staticmethod
    def match_static(event_log: pd.DataFrame, text_dataset: pd.DataFrame, strength: float) -> pd.DataFrame:
        """
        Match textual and eventlog datasets "static". Assign textual context data on case-level to the first activity per
        trace by most matching keywords per category. For a certain percentage, the other outcome is substituted with
        texts covering oposite topic group.
        :param event_log: DataFrame of eventlog
        :param text_dataset: DataFrame of text dataset
        :param strength of topic overlap
        :return: enriched DataFrame
        """

        split = [(0.5 + 0.1 * strength, 0.5 - 0.1 * strength), (0.5 - 0.1 * strength, 0.5 + 0.1 * strength),
                 (0.5 + 0.1 * strength, 0.5 - 0.1 * strength), (0.5 - 0.1 * strength, 0.5 + 0.1 * strength),
                 (0.5 + 0.1 * strength, 0.5 - 0.1 * strength), (0.5 - 0.1 * strength, 0.5 + 0.1 * strength)]

        desc_list = text_dataset["desc"].tolist()

        case_id_map = event_log.groupby('Case ID').tail(1).reset_index()[["Case ID", "label"]]
        neg_case_nr = case_id_map["label"].value_counts()[1]
        pos_case_nr = case_id_map["label"].value_counts()[0]

        occurances = []
        for topic in keyword_map_title:
            occurrence_list = DatasetEnrichment.count_topic_word_occurances(desc_list, keyword_map_title[topic])
            occurrence_list.sort(key=lambda x: x[1], reverse=True)
            occurances.append(occurrence_list)

        list_to_map_accepted = []
        list_to_map_rejected = []

        # Create number of desc per topic to accepted/rejected assignment

        for idx, topic in enumerate(keyword_map_title):
            number_per_topic = int(split[idx][1] * (neg_case_nr / 3) + split[idx][0] * (pos_case_nr / 3))
            list_to_map_accepted.append(int(split[idx][0] * (pos_case_nr / 3)))
            list_to_map_rejected.append(number_per_topic - int(split[idx][0] * (pos_case_nr / 3)))

        text_list_acc, text_list_rej = DatasetEnrichment.iterat_map_text(occurances, list_to_map_accepted,
                                                                         list_to_map_rejected)

        text_list_acc_flat = [item for sublist in text_list_acc for item in sublist]
        text_list_rej_flat = [item for sublist in text_list_rej for item in sublist]

        # Ensure that all cases have a text
        if len(text_list_rej_flat) > neg_case_nr:
            text_list_rej_flat = text_list_rej_flat[:neg_case_nr]
        if len(text_list_acc_flat) > pos_case_nr:
            text_list_acc_flat = text_list_acc_flat[:pos_case_nr]
        if len(text_list_acc_flat) < pos_case_nr:
            difference = pos_case_nr - len(text_list_acc_flat)
            for i in range(difference):
                text_list_acc_flat.append(occurances[0].pop())
        if len(text_list_rej_flat) < neg_case_nr:
            difference = neg_case_nr - len(text_list_rej_flat)
            for i in range(difference):
                text_list_rej_flat.append(occurances[0].pop())

        # Adding the columns to the existing dataset
        event_log["desc"] = ""
        event_log["title"] = ""
        event_log["emp_title"] = ""
        textdataset = text_dataset.reset_index()

        event_count = event_log.groupby(["Case ID"], sort=False).size()
        cases = event_log.groupby(["Case ID"], sort=False).last()["label"].tolist()
        sampled_text_match = []
        for idx, evt_count in enumerate(event_count):
            if cases[idx] == 1:
                indx = text_list_acc_flat.pop()[0]
                sampled_text_match.append([textdataset.iloc[indx]["desc"], textdataset.iloc[indx]["title"],
                                           textdataset.iloc[indx]["emp_title"]])
            else:
                indx = text_list_rej_flat.pop()[0]
                sampled_text_match.append([textdataset.iloc[indx]["desc"], textdataset.iloc[indx]["title"],
                                           textdataset.iloc[indx]["emp_title"]])
            for i in range(1, evt_count):
                sampled_text_match.append(["", "", ""])

        event_log = event_log.assign(text=sampled_text_match)
        split_text = pd.DataFrame(event_log["text"].tolist(), columns=["desc", "title", "emp_title"])
        event_log = pd.concat([event_log.drop(["text", "desc", "title", "emp_title"], axis=1), split_text], axis=1)
        return event_log

    @staticmethod
    def save_to_file(enriched_dataset: pd.DataFrame, file_path_out: str) -> None:
        """
        Save enriched dataset to .csv
        :param enriched_dataset: DataFrame with enriched dataset
        :param file_path_out: Filepath for .csv
        :return: None
        """

        enriched_dataset.to_csv(file_path_out, index=False, sep=";")

    @staticmethod
    def count_topic_word_occurances(desc_list, topic_keyword_list):
        """
        Count word occurrences per topic for a given set of texts
        :param desc_list:
        :param topic_keyword_list:
        :return:
        """

        occurrence_list = []
        for idx, text in enumerate(desc_list):
            c = Counter(''.join(char for char in s.lower() if char.isalpha()) for s in text.split())

            occurrence_list.append((idx, sum(c[v] for v in topic_keyword_list)))
        return occurrence_list

    @staticmethod
    def iterat_map_text(occurances: List[List[int]], list_to_map_accepted: List[int], list_to_map_rejected: List[int],
                        return_occurances=False) -> Tuple[List[int], List[int]]:
        """
        Assign descriptions to the individual outcomes according to the topics they cover
        :param occurances: occurrances of word per topic
        :param list_to_map_accepted: accepted outcomes
        :param list_to_map_rejected: rejected outcomess
        :param return_occurances: should occurance values be returned
        :return:
        """

        text_list_acc = [[], [], [], [], [], []]
        text_list_rej = [[], [], [], [], [], []]

        removed = 0
        while not ((all(v == 0 for v in list_to_map_accepted)) and (all(v == 0 for v in list_to_map_rejected))):
            for idx, topic in enumerate(keyword_map_title):
                num_to_remove_acc = 5
                num_to_remove_rej = 5
                topN_rej = []
                topN_acc = []
                if list_to_map_accepted[idx] < 5:
                    num_to_remove_acc = list_to_map_accepted[idx]

                if list_to_map_rejected[idx] < 5:
                    num_to_remove_rej = list_to_map_rejected[idx]

                if (num_to_remove_acc == 0) and (num_to_remove_rej == 0):
                    continue

                removed += num_to_remove_acc + num_to_remove_rej

                if num_to_remove_acc != 0:
                    list_to_map_accepted[idx] = list_to_map_accepted[idx] - num_to_remove_acc

                    topN_acc = occurances[idx][0:num_to_remove_acc]
                    del occurances[idx][0:num_to_remove_acc]
                    text_list_acc[idx].extend(topN_acc)

                if num_to_remove_rej != 0:
                    list_to_map_rejected[idx] = list_to_map_rejected[idx] - num_to_remove_rej

                    topN_rej = occurances[idx][0:num_to_remove_rej]
                    del occurances[idx][0:num_to_remove_rej]
                    text_list_rej[idx].extend(topN_rej)

                topN2 = topN_rej + topN_acc
                idx_list = [i[0] for i in topN2]
                idx_range = [*range(len(occurances))]
                idx_range.pop(idx)
                for lst in idx_range:
                    occurances[lst] = list(filter(lambda x: x[0] not in idx_list, occurances[lst]))
                    occurances[lst].sort(key=lambda x: x[1], reverse=True)

        if return_occurances:
            return text_list_acc, text_list_rej, occurances
        return text_list_acc, text_list_rej


if __name__ == '__main__':

    for i in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        DatasetEnrichment.build(config["data"]["bpi17_pa"], config["data"]["lending_club_acc"], strategy="impure",
                                impurity=i,
                                out_path=config["data"]["synthetic"])
