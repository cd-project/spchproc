import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
from hmmlearn.hmm import GMMHMM

import utils


def load_dataset(data_file_path: str, data_length_path: str):
    features, data_length = None, None
    with open(data_file_path, "rb") as sf:
        features = pickle.load(sf)

    with open(data_length_path, "rb") as sf:
        data_length = pickle.load(sf)

    return features, data_length


def load_model_dict(model_dict_path: str):
    model_dict = None
    with open(model_dict_path, "rb") as mf:
        model_dict = pickle.load(mf)

    return model_dict


def run_train(config: dict):
    trainset, train_length = load_dataset(
        config["input_train_data_file"],
        config["input_train_length_file"]
    )

    gmm_hmm_models = train_gmm_hmm(trainset, train_length, config["syllable_file_path"])

    # Save trained model
    model_folder_path = "./output/model"
    utils.create_folder(model_folder_path)
    with open(os.path.join(model_folder_path, f'{config["config_name"]}.pkl'), "wb") as mf:
        pickle.dump(gmm_hmm_models, mf)


def run_test(config: dict):
    testset, test_length = load_dataset(
        config["input_test_data_file"],
        config["input_test_length_file"]
    )

    gmm_hmm_models = load_model_dict(config["input_model_dict_file"])

    preds = []
    reals = []
    labels = ['sil', '1', 'tram', '4', 'muoi', '9', 'trieu', '3', '7', '8', 'nghin', '6', '5', 'lam', '2', 'tu',
              '0', 'mot', 'linh', 'm1']

    for label in testset.keys():
        features = testset[label]
        lengths = test_length[label]
        start_index = 0
        for i in range(len(lengths)):
            single_feature = features[start_index: start_index + lengths[i]]
            score_i = {}
            for model_label in gmm_hmm_models.keys():
                score_i[model_label] = gmm_hmm_models[model_label].score(single_feature)

            preds.append(max(score_i, key=score_i.get))
            reals.append(label)
            start_index += lengths[i]

    # Save result
    result_folder_path = "../output/result"
    utils.save_result(preds, reals, labels, result_folder_path, config)


def train_gmm_hmm(dataset: dict, data_length: dict, syllable_file_path: str):
    syllable_df = pd.read_csv(syllable_file_path, sep="\t", header=None)
    syllable_df.rename(columns={0: "label", 1: "n_syllables"}, inplace=True)
    gmm_hmm_models = {}

    # Define general config of model
    states_num = 5
    n_mix = 3
    tmp_p = 1.0 / n_mix
    transmat_prior = np.array([[tmp_p, tmp_p, tmp_p, 0, 0],
                               [0, tmp_p, tmp_p, tmp_p, 0],
                               [0, 0, tmp_p, tmp_p, tmp_p],
                               [0, 0, 0, 0.5, 0.5],
                               [0, 0, 0, 0, 1]], dtype=float)

    startprob_prior = np.array([0.5, 0.5, 0, 0, 0], dtype=float)

    for label in dataset.keys():
        model = GMMHMM(n_components=states_num, n_mix=n_mix,
                       transmat_prior=transmat_prior, startprob_prior=startprob_prior,
                       covariance_type='diag', n_iter=10, random_state=42)
        train_data = dataset[label]
        length = data_length[label]

        model.fit(train_data, lengths=length)  # get optimal parameters
        gmm_hmm_models[label] = model

    return gmm_hmm_models


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Commandline Gaussian mixture for HMM acoustic model.")
    parser.add_argument("--train_config_file", help="Path to training config file")
    parser.add_argument("--test_config_file", help="Path to testing config file")

    args = parser.parse_args()

    if args.train_config_file:
        with open(args.train_config_file) as f:
            run_train_config = json.load(f)
            run_train(run_train_config)

    if args.test_config_file:
        with open(args.test_config_file) as f:
            run_test_config = json.load(f)
            run_test(run_test_config)
