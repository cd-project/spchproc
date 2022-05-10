import os

import pandas as pd
from sklearn import metrics


def create_folder(folder_path):
    folder_path = os.path.abspath(folder_path)
    project_folder = os.path.dirname(os.getcwd())

    if os.path.commonpath([folder_path, project_folder]) != project_folder:
        return -1

    if folder_path == project_folder:
        return 0

    if os.path.isdir(folder_path):
        return 0

    create_folder(os.path.dirname(folder_path))
    os.mkdir(folder_path)

    return 0


def save_result(preds, reals, labels, result_folder_path, config):
    # Save result
    create_folder(result_folder_path)

    pred_real_df = pd.DataFrame(list(zip(reals, preds)), columns=["real_label", "predict_label"])
    pred_real_df.to_csv(
        os.path.join(result_folder_path, f'predict_real_label_{config["config_name"]}.csv'),
        index=False)

    clsf_report = pd.DataFrame(metrics.classification_report(reals, preds, labels=labels, output_dict=True)).transpose()
    # print(clsf_report)
    # Save testing report
    clsf_report.to_csv(os.path.join(result_folder_path, f'test_metrics_{config["config_name"]}.csv'),
                       index=True)
