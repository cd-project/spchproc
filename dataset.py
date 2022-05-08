import glob
import os

import librosa
import numpy as np
import pandas as pd


def get_mfcc_data_series_features(x, sound, sr, n_mfcc, hop_length, n_fft, delta_width):
    s = np.floor(x.loc["start"] * sr).astype(int)
    e = np.ceil(x.loc["end"] * sr).astype(int)
    sound = sound[s:e]

    # Generate mfcc features (mfcc and 2 derivation)
    mfcc = librosa.feature.mfcc(y=sound, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
    delta_mfcc = librosa.feature.delta(mfcc, width=delta_width)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2, width=delta_width)

    return np.concatenate((mfcc, delta_mfcc, delta2_mfcc)).T


def normalize_mfcc_features(mfcc_features):
    min_features = np.min(mfcc_features, axis=0)
    max_features = np.max(mfcc_features, axis=0)
    return (mfcc_features - min_features) / (max_features - min_features)


def get_dataset_df(data_dirs, sr, n_mfcc, hop_length, n_fft, delta_width):
    df = pd.DataFrame(columns=["fid", "start", "end", "label", "mfcc_features"])
    for data_dir in data_dirs:
        fids = [os.path.basename(p).split(".")[0] for p in glob.glob(os.path.join(data_dir, "*.txt"))]
        for fid in fids:
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

    return df


def get_template_df(df: pd.DataFrame, n_sample_per_word: int):
    df_template = pd.DataFrame(columns=df.columns)
    for label in df["label"].unique():
        df_template = pd.concat(
            [df_template, df[df["label"] == label].sample(n=n_sample_per_word, replace=False, random_state=42)], axis=0,
            ignore_index=True)

    return df_template
