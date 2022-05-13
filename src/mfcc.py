import argparse
import glob
import json
import os
import pickle

import librosa
import numpy as np
import pandas as pd

import utils


def get_mfcc_data_series_features(x: pd.Series, sound: np.ndarray, sr: int, n_mfcc: int, hop_length: int, n_fft: int,
                                  delta_width: int):
    """

    :param a row of dataframe which contain (start, end, mfcc_features) columns x:
    :param sound: (n_samples, ) array
    :param sampling rate sr:
    :param dimension of each mfcc feature vector n_mfcc:
    :param hop_length:
    :param n_fft:
    :param delta_width:
    :return:
    """

    s = np.floor(x.loc["start"] * sr).astype(int)
    e = np.ceil(x.loc["end"] * sr).astype(int)
    sound = sound[s:e]

    # Generate mfcc features (mfcc and 2 derivation)
    mfcc = librosa.feature.mfcc(y=sound, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
    delta_mfcc = librosa.feature.delta(mfcc, width=delta_width, mode="nearest")
    delta2_mfcc = librosa.feature.delta(mfcc, order=2, width=delta_width, mode="nearest")

    return np.concatenate((mfcc, delta_mfcc, delta2_mfcc)).T


def normalize_mfcc_features(mfcc_features):
    min_features = np.min(mfcc_features, axis=0)
    max_features = np.max(mfcc_features, axis=0)
    return (mfcc_features - min_features) / (max_features - min_features)


def get_dataset_df(data_dirs: list, sr: int, n_mfcc: int, hop_length: int, n_fft: int, delta_width: int):
    """

    :param list of data directories which contain sound and label file data_dirs:
    :param sampling rate use for sampling wave form sr:
    :param expected n_dimension of mfcc feature n_mfcc:
    :param step between each frame use for extract mfcc feature hop_length:
    :param n_fft:
    :param delta_width:
    :return:
    """

    df = pd.DataFrame(columns=["fid", "start", "end", "label", "mfcc_features"])
    for data_dir in data_dirs:
        fids = [os.path.basename(p).split(".")[0] for p in glob.glob(os.path.join(data_dir, "*.txt"))]
        
        for fid in fids:
            print(fid)
            # Get data from label file
            label_file_path = os.path.join(data_dir, f"{fid}.txt")
            dfi = pd.read_csv(label_file_path, sep="\t", header=None)
            dfi.rename(columns={0: "start", 1: "end", 2: "label"}, inplace=True)
            dfi["fid"] = [fid] * len(dfi)

            # Create mfcc feature with each label in data frame of fid file
            sound_file_path = os.path.join(data_dir, f"{fid}.wav")
            sound_i, _ = librosa.load(sound_file_path, sr=sr)
            dfi["mfcc_features"] = dfi.apply(get_mfcc_data_series_features,
                                             args=(sound_i, sr, n_mfcc, hop_length, n_fft, delta_width), axis=1)
            dfi["mfcc_features"] = dfi["mfcc_features"].apply(lambda x: normalize_mfcc_features(x))

            df = pd.concat([df, dfi], axis=0, ignore_index=True)
            df.drop(columns=["fid", "start", "end"], inplace=True)

        # Remove nan data
        remove_indexes = []
        for i in range(len(df)):
            if np.isnan(np.sum(df.iloc[i]["mfcc_features"])):
                remove_indexes.append(i)

        df.drop(df.index[remove_indexes], inplace=True, axis=0)
    # print("ok")
    return df


def get_template_df(df: pd.DataFrame, wsa_num: int):
    df_template = pd.DataFrame(columns=df.columns)
    for label in df["label"].unique():
        df_template = pd.concat(
            [df_template, df[df["label"] == label].sample(n=wsa_num, replace=False, random_state=42)], axis=0,
            ignore_index=True)

    return df_template


def get_dataset_for_hmm_from_df(df: pd.DataFrame):
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


def save_data_for_hmm(config: dict, save_folder_path):
    utils.create_folder(save_folder_path)

    df = get_dataset_df(config["data_dirs"], config["sr"], config["n_mfcc"], config["hop_length"],
                        config["n_fft"], config["delta_width"])

    dataset, data_length = get_dataset_for_hmm_from_df(df)

    with open(os.path.join(save_folder_path, f'{config["config_name"]}_set'), "wb") as sf:
        pickle.dump(dataset, sf)

    with open(os.path.join(save_folder_path, f'{config["config_name"]}_length'), "wb") as sf:
        pickle.dump(data_length, sf)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--hmm_train_create", help="Path to config file of hmm training data")
    parser.add_argument("--hmm_test_create", help="Path to config file of hmm testing data")

    args = parser.parse_args()

    if args.hmm_train_create:
        save_train_data_folder = "../output/data/train_data"
        with open(args.hmm_train_create) as f:
            train_data_config = json.load(f)
            save_data_for_hmm(train_data_config, save_train_data_folder)

    if args.hmm_test_create:
        save_test_data_folder = "../output/data/test_data"
        with open(args.hmm_test_create) as f:
            test_data_config = json.load(f)
            save_data_for_hmm(test_data_config, save_test_data_folder)
