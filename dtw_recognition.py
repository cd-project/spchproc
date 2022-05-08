import copy

import numpy as np
import pandas as pd

from dtw import dtw


def predict(template_df, test_mfcc_features):
    preds = []
    for mfcc_test_i in test_mfcc_features:
        dt_list_i = []

        for j in range(len(template_df)):
            mfccj = template_df.iloc[j]["mfcc_features"]
            # c_j = template_df.iloc[j]["conv"]
            # dist_list = [lambda x, y: (x - y).T.dot(np.linalg.pinv(c_j[l])).dot(x - y) for l in range(len(c_j))]
            dist_list = [lambda x, y: np.linalg.norm(x - y, ord=2)] * len(mfccj)

            dtj = dtw(mfccj, mfcc_test_i, dist_list)[0]
            dt_list_i.append(dtj)

        preds.append(template_df.iloc[np.argmin(dt_list_i)]["label"])

    return preds


def segmental_k_means(mfcc_features, n_state):
    n_templates, n_mfcc = len(mfcc_features), mfcc_features[0].shape[1]
    new_feature = np.zeros((n_state, n_mfcc))
    new_cov = np.zeros((n_state, n_mfcc, n_mfcc))

    # Uniform initialize boundary between state
    boundaries = []
    for j in range(n_templates):
        nj = mfcc_features[j].shape[0]
        boundary_j = np.arange(0, nj, nj // n_state)
        boundary_j = list(boundary_j)
        while len(boundary_j) >= n_state + 1:
            boundary_j.pop()
        boundary_j.append(nj)
        boundaries.append(boundary_j)

    while True:
        prev_features = copy.deepcopy(new_feature)
        for i in range(n_state):
            vector_list = np.zeros((1, n_mfcc))
            for j in range(n_templates):
                s = boundaries[j][i]
                e = boundaries[j][i + 1]
                vector_list = np.concatenate((vector_list, mfcc_features[j][s:e]), axis=0)

            new_feature[i] = np.mean(vector_list[1:], axis=0)
            if len(vector_list[1:]) <= 1:
                new_cov[i] = np.eye(n_mfcc)
            else:
                new_cov[i] = np.cov(vector_list[1:].T) * (len(vector_list[1:]) - 1) / len(vector_list[1:])

            # Diag covariance matrix
            new_cov[i] = np.diag(np.diag(new_cov[i]))

        for j in range(n_templates):
            dist_list = [lambda x, y: (x - y).T.dot(np.linalg.pinv(new_cov[l])).dot(x - y) for l in range(n_state)]
            _, path_t, path_s = dtw(new_feature, mfcc_features[j], dist_list)

            current_segment = 1
            for k in range(1, len(path_s)):
                if path_t[k] != path_t[k - 1]:
                    boundaries[j][current_segment] = k
                    current_segment += 1

        if np.sqrt(np.linalg.norm(prev_features - new_feature, ord="fro")) < 1e-5:
            break

    return new_feature, new_cov


def reduce_models(template_df, syllables_file_path):
    syllable_df = pd.read_csv(syllables_file_path, sep="\t", header=None)
    syllable_df.rename(columns={0: "label", 1: "n_syllables"}, inplace=True)

    new_df = pd.DataFrame(columns=list(template_df.columns) + ["conv"])
    for label in template_df["label"].unique():
        mfcc_features = list(template_df[template_df["label"] == label]["mfcc_features"])
        n_state = syllable_df["n_syllables"][syllable_df["label"] == label].iloc[0]

        reduced_mfcc_feature, new_cov = segmental_k_means(mfcc_features, n_state)
        new_df.loc[len(new_df)] = [label, reduced_mfcc_feature, new_cov]

    return new_df
