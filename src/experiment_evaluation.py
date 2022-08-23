from utils import *

config = toml.load("../config.toml")

base_path = config["result"]["results"]

from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from scipy.stats import spearmanr

strengths = ["impure_0.0", "impure_0.1", "impure_0.2", "impure_0.3", "impure_0.4", "impure_0.5", "impure_0.6",
             "impure_0.7", "impure_0.8", "impure_0.9", "impure_1.0"]
num_training_samples = 5000
num_eval_samples = 5000
num_explanations = 500


def model_metrics_calculation() -> None:
    """
    Consolidate and calculate model metrics
    :return: None
    """

    ###################################
    # Baseline
    ###################################

    strategy = "strategy 1/"

    f1_scores = []
    roc_auc_scores = []

    for s in strengths:
        path_class_proba_xgb = base_path + strategy + "1_xgb_class_proba_" + s + ".pckl"
        path_y_test = base_path + strategy + "1_xgb_y_test_" + s + ".pckl"

        with open(path_class_proba_xgb, 'rb') as f:
            proba_xgb = pickle.load(f)
        with open(path_y_test, 'rb') as f:
            y_test = pickle.load(f)

        labels_xgb = np.argmax(proba_xgb, axis=1)
        labels_xgb = labels_xgb.tolist()

        f1_scores.append([s, f1_score(y_test.tolist(), labels_xgb)])
        roc_auc_scores.append([s, roc_auc_score(y_test.tolist(), labels_xgb)])
    pickl_save(base_path + "results_paper/b_f1_score.pckl", f1_scores)
    pickl_save(base_path + "results_paper/b_auc_roc.pckl", roc_auc_scores)

    ###################################
    # Strategy 1
    ###################################

    strategy = "strategy 1/"

    f1_scores = []
    roc_auc_scores = []
    f1_scores_brt = []
    roc_auc_scores_brt = []

    for s in strengths:
        path_logits_bert = base_path + strategy + "1_bert_logits_" + s + ".pckl"
        path_class_proba_xgb = base_path + strategy + "1_xgb_class_proba_" + s + ".pckl"
        path_y_test = base_path + strategy + "1_xgb_y_test_" + s + ".pckl"

        with open(path_logits_bert, 'rb') as f:
            logits_bert = pickle.load(f)
        with open(path_class_proba_xgb, 'rb') as f:
            proba_xgb = pickle.load(f)
        with open(path_y_test, 'rb') as f:
            y_test = pickle.load(f)

        labels_combined, labels_bert, labels_xgb = calculate_prediction_strat1(strategy=1, logits_1=logits_bert,
                                                                               class_proba_2=proba_xgb)
        pickl_save(base_path + "results_paper/" + strategy + "1_labels_xgb_" + s + ".pckl",
                   [y_test.tolist(), labels_xgb])
        pickl_save(base_path + "results_paper/" + strategy + "1_labels_bert_" + s + ".pckl",
                   [y_test.tolist(), labels_bert])
        pickl_save(base_path + "results_paper/" + strategy + "1_labels_combined_" + s + ".pckl",
                   [y_test.tolist(), labels_combined])
        f1_scores.append([s, f1_score(y_test.tolist(), labels_combined)])
        roc_auc_scores.append([s, roc_auc_score(y_test.tolist(), labels_combined)])
        f1_scores_brt.append([s, f1_score(y_test.tolist(), labels_bert)])
        roc_auc_scores_brt.append([s, roc_auc_score(y_test.tolist(), labels_bert)])
    pickl_save(base_path + "results_paper/1_f1_score.pckl", f1_scores)
    pickl_save(base_path + "results_paper/brt_f1_score.pckl", f1_scores_brt)
    pickl_save(base_path + "results_paper/brt_auc_roc.pckl", roc_auc_scores_brt)
    pickl_save(base_path + "results_paper/1_f1_score.pckl", f1_scores)
    pickl_save(base_path + "results_paper/1_auc_roc.pckl", roc_auc_scores)

    ###################################
    # Strategy 2
    ###################################

    strategy = "strategy 2/"

    f1_scores = []
    roc_auc_scores = []

    for s in strengths:
        path_class_proba_xgb = base_path + strategy + "2_xgb_class_proba_" + s + ".pckl"
        path_y_test = base_path + "strategy 1/1_xgb_y_test_" + s + ".pckl"

        with open(path_class_proba_xgb, 'rb') as f:
            proba_xgb = pickle.load(f)
        with open(path_y_test, 'rb') as f:
            y_test = pickle.load(f)

        labels_xgb = np.argmax(proba_xgb, axis=1)
        labels_xgb = labels_xgb.tolist()

        pickl_save(base_path + "results_paper/" + strategy + "2_labels_xgb_" + s + ".pckl",
                   [y_test.tolist(), labels_xgb])
        f1_scores.append([s, f1_score(y_test.tolist(), labels_xgb)])
        roc_auc_scores.append([s, roc_auc_score(y_test.tolist(), labels_xgb)])
    pickl_save(base_path + "results_paper/2_f1_score.pckl", f1_scores)
    pickl_save(base_path + "results_paper/2_auc_roc.pckl", roc_auc_scores)


def rediscover_rate_calculation() -> None:
    """
    Calculate Rediscovery Rate
    :return: None
    """

    ###################################
    # Strategy 1
    ###################################

    strategy = "strategy 1/"
    results_strategy_1 = []
    for s in strengths:
        path_shap_bert = base_path + strategy + "1_bert_shap_values_" + s + ".pckl"
        with open(path_shap_bert, 'rb') as f:
            shap_bert = pickle.load(f)
        rediscovery_score = calculate_rediscovery_metric(shap_values=shap_bert)
        results_strategy_1.append(["strategy 1", s, rediscovery_score])
    pickl_save(base_path + "results_paper/" + strategy + "rediscovery_score.pckl", results_strategy_1)

    ###################################
    # Strategy 2
    ###################################

    strategy = "strategy 2/"
    results_strategy_2 = []
    for s in strengths:
        path_shap_xgb = base_path + strategy + "2_xgb_shap_values_" + s + ".pckl"
        with open(path_shap_xgb, 'rb') as f:
            shap_xgb = pickle.load(f)
        rediscovery_score = calculate_rediscovery_metric(shap_values=shap_xgb)
        results_strategy_2.append(["strategy 1", s, rediscovery_score])
        print(rediscovery_score)
    pickl_save(base_path + "results_paper/" + strategy + "rediscovery_score.pckl", results_strategy_2)


def parsimony_calculation() -> None:
    """
    Calculate Parsimony Score
    :return: None
    """

    ###################################
    # Baseline
    ###################################

    strategy = "strategy 1/"
    results_parsimony = []
    for s in strengths:
        path_shap_xgb = base_path + strategy + "1_xgb_shap_values_" + s + ".pckl"
        with open(path_shap_xgb, 'rb') as f:
            shap_xgb = pickle.load(f)
        feature_importances = get_n_important_features(shap_values=shap_xgb, n=None, remove_stop_words=False,
                                                       with_importance=True)
        df = pd.DataFrame(feature_importances, ["feature_name", "importance"]).T
        df["importance"] = pd.to_numeric(df["importance"])
        results_parsimony.append([len(df[df['importance'] == 0]), len(df[df['importance'] != 0])])
    pickl_save(base_path + "results_paper/b_parsimony.pckl", results_parsimony)

    ###################################
    # Strategy 1
    ###################################

    strategy = "strategy 1/"
    results_parsimony = []
    for s in strengths:
        path_shap_xgb = base_path + strategy + "1_xgb_shap_values_" + s + ".pckl"
        with open(path_shap_xgb, 'rb') as f:
            shap_xgb = pickle.load(f)
        path_shap_bert = base_path + strategy + "1_bert_shap_values_" + s + ".pckl"
        with open(path_shap_bert, 'rb') as f:
            shap_bert = pickle.load(f)
        feature_importances_xgb = get_n_important_features(shap_values=shap_xgb, n=None, remove_stop_words=False,
                                                           with_importance=True)
        feature_importances_bert = get_n_important_features(shap_values=shap_bert, n=None, remove_stop_words=False,
                                                            with_importance=True)
        feature_importances = (feature_importances_bert[0] + feature_importances_xgb[0],
                               feature_importances_bert[1] + feature_importances_xgb[1])

        df = pd.DataFrame(feature_importances, ["feature_name", "importance"]).T
        df["importance"] = pd.to_numeric(df["importance"])
        results_parsimony.append([len(df[df['importance'] == 0]), len(df[df['importance'] != 0])])
    pickl_save(base_path + "results_paper/1_parsimony.pckl", results_parsimony)

    ###################################
    # Strategy 2
    ###################################

    strategy = "strategy 2/"
    results_parsimony = []
    for s in strengths:
        path_shap_xgb = base_path + strategy + "2_xgb_shap_values_" + s + ".pckl"
        with open(path_shap_xgb, 'rb') as f:
            shap_xgb = pickle.load(f)
        feature_importances = get_n_important_features(shap_values=shap_xgb, n=None, remove_stop_words=False,
                                                       with_importance=True)
        df = pd.DataFrame(feature_importances, ["feature_name", "importance"]).T
        df["importance"] = pd.to_numeric(df["importance"])
        results_parsimony.append([len(df[df['importance'] == 0]), len(df[df['importance'] != 0])])
    pickl_save(base_path + "results_paper/2_parsimony.pckl", results_parsimony)


def monotonicity_caluclation() -> None:
    """
    Calculate Monotonicity Score
    :return: None
    """

    ###################################
    # Baseline
    ###################################

    monotonicity = []
    for s in strengths:
        task_feature_importance_path = base_path + "strategy 1/1_xgb_feature_importances_" + s + ".pckl"
        with open(task_feature_importance_path, 'rb') as f:
            task_feature_importance = pickle.load(f)
        task_feature_names_path = base_path + "strategy 1/1_xgb_feature_names_" + s + ".pckl"
        with open(task_feature_names_path, 'rb') as f:
            task_feature_names = pickle.load(f)

        shap_feature_importance_path = base_path + "strategy 1/1_xgb_shap_values_" + s + ".pckl"
        with open(shap_feature_importance_path, 'rb') as f:
            shap_feature_importance = pickle.load(f)

        feature_importances_shap = get_n_important_features(shap_feature_importance, n=None, with_importance=True)
        df_shap = pd.DataFrame(feature_importances_shap, ["feature_name", "importance"]).T
        df_xgb = pd.DataFrame([task_feature_names, task_feature_importance], ["feature_name", "importance_task"]).T
        df_importance = pd.merge(df_shap, df_xgb, how='inner', on="feature_name")
        df_importance.sort_values(by='importance', ascending=False, inplace=True)
        coef, p = spearmanr(df_importance['importance'], df_importance['importance_task'])
        monotonicity.append(coef)

    pickl_save(base_path + "results_paper/b_monotonicity.pckl", monotonicity)

    ###################################
    # Strategy 2
    ###################################

    monotonicity = []
    for s in strengths:
        task_feature_importance_path = base_path + "strategy 2/2_xgb_feature_importances_" + s + ".pckl"
        with open(task_feature_importance_path, 'rb') as f:
            task_feature_importance = pickle.load(f)
        task_feature_names_path = base_path + "strategy 2/2_xgb_feature_names_" + s + ".pckl"
        with open(task_feature_names_path, 'rb') as f:
            task_feature_names = pickle.load(f)

        shap_feature_importance_path = base_path + "strategy 2/2_xgb_shap_values_" + s + ".pckl"
        with open(shap_feature_importance_path, 'rb') as f:
            shap_feature_importance = pickle.load(f)

        feature_importances_shap = get_n_important_features(shap_feature_importance, n=None, with_importance=True)
        df_shap = pd.DataFrame(feature_importances_shap, ["feature_name", "importance"]).T
        df_xgb = pd.DataFrame([task_feature_names, task_feature_importance], ["feature_name", "importance_task"]).T
        df_importance = pd.merge(df_shap, df_xgb, how='inner', on="feature_name")
        df_importance.sort_values(by='importance', ascending=False, inplace=True)
        coef, p = spearmanr(df_importance['importance'], df_importance['importance_task'])
        monotonicity.append(coef)
    pickl_save(base_path + "results_paper/2_monotonicity.pckl", monotonicity)


def model_time_calculation() -> None:
    """
    Caclulate model time for training and explanation
    :return: None
    """

    ###################################
    # Baseline
    ###################################

    strategy = "strategy 1/"
    training = []
    explanation = []

    for s in strengths:
        path_time_xgb = base_path + strategy + "1_xgb_time_" + s + ".pckl"
        with open(path_time_xgb, 'rb') as f:
            time_xgb = pickle.load(f)
        training.append(time_xgb[2])
        explanation.append(time_xgb[4])
    pickl_save(base_path + "results_paper/b_training_times.pckl", training)
    pickl_save(base_path + "results_paper/b_explanation_times.pckl", explanation)

    ###################################
    # Strategy 1
    ###################################

    strategy = "strategy 1/"
    training = []
    explanation = []

    for s in strengths:
        path_time_bert = base_path + strategy + "1_bert_time_" + s + ".pckl"
        with open(path_time_bert, 'rb') as f:
            time_bert = pickle.load(f)

        path_time_xgb = base_path + strategy + "1_xgb_time_" + s + ".pckl"
        with open(path_time_xgb, 'rb') as f:
            time_xgb = pickle.load(f)
        training.append(max(time_xgb[2], time_bert[2]))
        explanation.append(max(time_xgb[4], time_bert[4]))
    pickl_save(base_path + "results_paper/1_training_times.pckl", training)
    pickl_save(base_path + "results_paper/1_explanation_times.pckl", explanation)

    ###################################
    # Strategy 2
    ###################################

    strategy = "strategy 2/"
    training = []
    explanation = []

    for s in strengths:
        path_time_bert = base_path + "strategy 1/1_bert_time_" + s + ".pckl"
        with open(path_time_bert, 'rb') as f:
            time_bert = pickle.load(f)

        path_time_xgb = base_path + strategy + "2_xgb_time_" + s + ".pckl"
        with open(path_time_xgb, 'rb') as f:
            time_xgb = pickle.load(f)
        training.append(time_xgb[2] + time_bert[2])
        explanation.append(time_xgb[4] + time_bert[4])
    pickl_save(base_path + "results_paper/2_training_times.pckl", training)
    pickl_save(base_path + "results_paper/2_explanation_times.pckl", explanation)


if __name__ == '__main__':
    model_metrics_calculation()
    parsimony_calculation()
    rediscover_rate_calculation()
    model_time_calculation()
    monotonicity_caluclation()
