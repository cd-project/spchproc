import os.path
import pickle

import numpy as np
import pandas as pd
from hmmlearn.hmm import GMMHMM
from sklearn.utils import shuffle

import dataset


def get_dataset(df):
    my_data = {}
    data_length = {}

    for i in range(len(df)):
        label = df.iloc[i]["label"]
        features = df.iloc[i]["mfcc_features"].tolist()
        if label not in my_data.keys():
            data_length[label] = []
            my_data[label] = []

        data_length[label].append(len(features))
        my_data[label] += features

    return my_data, data_length


def load_dataset(data_file_path, data_length_path):
    features, data_length = None, None
    with open(data_file_path, "rb") as sf:
        features = pickle.load(sf)

    with open(data_length_path, "rb") as sf:
        data_length = pickle.load(sf)

    return features, data_length


def load_model_dict(model_dict_path):
    model_dict = None
    with open(model_dict_path, "rb") as mf:
        model_dict = pickle.load(mf)

    return model_dict


def create_train_test_data(config):
    df = dataset.get_dataset_df(config["data_dirs"], config["sr"], config["n_mfcc"], config["hop_length"],
                                config["n_fft"], config["delta_width"])

    df = shuffle(df)
    train_size = int(config["split_point"] * len(df))
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]

    testset, test_length = get_dataset(test_df)
    trainset, train_length = get_dataset(train_df)

    with open(os.path.join(config["save_data_dir"], config["save_train_data_file"]), "wb") as sf:
        pickle.dump(trainset, sf)

    with open(os.path.join(config["save_data_dir"], config["save_train_length_file"]), "wb") as sf:
        pickle.dump(train_length, sf)

    with open(os.path.join(config["save_data_dir"], config["save_test_data_file"]), "wb") as sf:
        pickle.dump(testset, sf)

    with open(os.path.join(config["save_data_dir"], config["save_test_length_file"]), "wb") as sf:
        pickle.dump(test_length, sf)


def run_train(config):
    trainset, train_length = load_dataset(
        os.path.join(config["input_data_dir"], config["input_train_data_file"]),
        os.path.join(config["input_data_dir"], config["input_train_length_file"])
    )

    gmm_hmm_models = train_gmm_hmm(trainset, train_length, config["syllable_file_path"])
    with open(os.path.join(config["save_model_dir"], config["save_model_dict_file"]), "wb") as mf:
        pickle.dump(gmm_hmm_models, mf)


def run_test(config):
    testset, test_length = load_dataset(
        os.path.join(config["input_data_dir"], config["input_test_data_file"]),
        os.path.join(config["input_data_dir"], config["input_test_length_file"])
    )

    gmm_hmm_models = load_model_dict(os.path.join(config["input_model_dir"], config["input_model_dict_file"]))

    preds = []
    reals = []

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

    print(np.sum(np.array(preds) == np.array(reals)) / len(preds))


def train_gmm_hmm(dataset, data_length, syllable_file_path):
    syllable_df = pd.read_csv(syllable_file_path, sep="\t", header=None)
    syllable_df.rename(columns={0: "label", 1: "n_syllables"}, inplace=True)
    gmm_hmm_models = {}

    # Define general config of model
    states_num = 5
    n_mix = 3
    tmp_p = 1.0 / (states_num - 2)
    transmat_prior = np.array([[tmp_p, tmp_p, tmp_p, 0, 0],
                               [0, tmp_p, tmp_p, tmp_p, 0],
                               [0, 0, tmp_p, tmp_p, tmp_p],
                               [0, 0, 0, 0.5, 0.5],
                               [0, 0, 0, 0, 1]], dtype=float)

    startprob_prior = np.array([0.5, 0.5, 0, 0, 0], dtype=float)

    for label in dataset.keys():
        model = GMMHMM(n_components=states_num, n_mix=n_mix,
                       transmat_prior=transmat_prior, startprob_prior=startprob_prior,
                       covariance_type='diag', n_iter=10)
        trainData = dataset[label]
        length = data_length[label]

        model.fit(trainData, lengths=length)  # get optimal parameters
        gmm_hmm_models[label] = model

    return gmm_hmm_models


if __name__ == '__main__':
    data_config = {
        "data_dirs": [
            "./data/05/19021396_PhamThanhVinh",
            "./data/05/19021372_BuiVanToan",
            "./data/05/19021381_NguyenVanTu",
            "../data/05/19021384_NguyenManhTuan"
        ],

        "save_data_dir": "./output/data",
        "save_train_data_file": "trainset-5:8:2022:22:57",
        "save_train_length_file": "train_length-5:8:2022:22:57",
        "save_test_data_file": "testset-5:8:2022:22:57",
        "save_test_length_file": "test_length-5:8:2022:22:57",

        "sr": 22050,
        "n_mfcc": 13,
        "hop_length": 256,
        "n_fft": 512,
        "delta_width": 5,
        "n_sample_per_word": 3,
        "split_point": 0.9
    }
    train_config = {
        "syllable_file_path": "./data/syllables.csv",

        "input_data_dir": "./output/data",
        "input_train_data_file": "trainset-5:8:2022:22:57",
        "input_train_length_file": "train_length-5:8:2022:22:57",

        "save_model_dir": "./output/model",
        "save_model_dict_file": "model_dict-5:8:2022:22:57.pkl",
    }
    test_config = {
        "input_data_dir": "./output/data",
        "input_test_data_file": "testset-5:8:2022:22:57",
        "input_test_length_file": "test_length-5:8:2022:22:57",

        "input_model_dir": "./output/model",
        "input_model_dict_file": "model_dict-5:8:2022:22:57.pkl"
    }

    create_train_test_data(data_config)
    run_train(train_config)
    run_test(test_config)
