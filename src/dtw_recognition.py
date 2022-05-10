import copy

from dataset import *
from dtw import dtw


def predict(template_df: pd.DataFrame, test_mfcc_features: list, warp):
    """

    :param list of available templates use for comparing with  template_df: (label, mfcc_features) columns
    :param mfcc features corresponding words test_mfcc_features: (label, mfcc_features) columns
    :param allowed warping step warp:
    :return: list of label presenting word for each testing input mfcc features
    """

    preds = []
    for mfcc_test_i in test_mfcc_features:
        dt_list_i = []

        for j in range(len(template_df)):
            mfccj = template_df.iloc[j]["mfcc_features"]
            c_j = template_df.iloc[j]["conv"]
            # dist_list = [lambda x, y: (x - y).T.dot(np.linalg.pinv(c_j[l])).dot(x - y) for l in range(len(mfccj))]
            dist_list = [lambda x, y: np.linalg.norm(x - y, ord=2)] * len(mfccj)

            dtj, _, _ = dtw(mfccj, mfcc_test_i, dist_list, warp=warp)
            dt_list_i.append(dtj)

        preds.append(template_df.iloc[np.argmin(dt_list_i)]["label"])

    return preds


def segmental_k_means(mfcc_features: list, n_state: int):
    """

    :param mfcc_feature of multi template which present the same word mfcc_features: (n_template, n_frame, n_mfcc) array
    :param expected number of segment of corresponding label n_state: int
    :return: new mfcc feature: (n_state, n_mfcc) array
    """

    n_templates, n_mfcc = len(mfcc_features), mfcc_features[0].shape[1]
    new_feature = np.zeros((n_state, n_mfcc))
    new_cov = np.zeros((n_state, n_mfcc, n_mfcc))

    masks = []
    for j in range(n_templates):
        nj = mfcc_features[j].shape[0]
        boundary_j = np.arange(0, nj, nj // n_state)
        boundary_j = list(boundary_j)
        while len(boundary_j) >= n_state + 1:
            boundary_j.pop()
        boundary_j.append(nj)

        mask_i = np.zeros(nj)
        for k in range(1, len(boundary_j)):
            mask_i[boundary_j[k - 1]: boundary_j[k]] = k - 1
        masks.append(mask_i)

    while True:
        prev_features = copy.deepcopy(new_feature)
        for i in range(n_state):
            vector_list = np.zeros((1, n_mfcc))
            for j in range(n_templates):
                vector_list = np.concatenate((vector_list, mfcc_features[j][masks[j] == i]), axis=0)

            if len(vector_list[1:]) <= 1:
                new_feature[i] = np.zeros(n_mfcc)
                new_cov[i] = np.eye(n_mfcc)
            else:
                new_feature[i] = np.mean(vector_list[1:], axis=0)
                new_cov[i] = np.cov(vector_list[1:].T) * (len(vector_list[1:]) - 1) / len(vector_list[1:])

            # Diag covariance matrix
            new_cov[i] = np.diag(np.diag(new_cov[i]))

        for j in range(n_templates):
            dist_list = [lambda x, y: (x - y).T.dot(np.linalg.pinv(new_cov[l])).dot(x - y) for l in range(n_state)]
            _, path_t, _ = dtw(new_feature, mfcc_features[j], dist_list)
            masks[j] = np.array(path_t[1:]) - 1

        if np.sqrt(np.linalg.norm(prev_features - new_feature, ord="fro")) < 1e-5:
            break

    return new_feature, new_cov


def reduce_models(template_df: pd.DataFrame, syllables_file_path: str):
    """

    :param template data frame of consist of mfcc_features of all word in dictionary template_df: (label, mfcc_features) columns
    :param path to predefined syllables for each word in dictionary syllables_file_path: str
    :return: new dataframe with reduced mfcc features: (label, new_mfcc_feature) columns
    """

    syllable_df = pd.read_csv(syllables_file_path, sep="\t", header=None)
    syllable_df.rename(columns={0: "label", 1: "n_syllables"}, inplace=True)

    new_df = pd.DataFrame(columns=list(template_df.columns) + ["conv"])
    for label in template_df["label"].unique():
        mfcc_features = list(template_df[template_df["label"] == label]["mfcc_features"])
        n_state = syllable_df["n_syllables"][syllable_df["label"] == label].iloc[0]

        reduced_mfcc_feature, new_cov = segmental_k_means(mfcc_features, n_state)
        new_df.loc[len(new_df)] = [label, reduced_mfcc_feature, new_cov]

    return new_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Commandline for DTW recognition.")
    parser.add_argument("--train_config_file", help="Path to train config file path")
    parser.add_argument("--test_config_file", help="Path to test config file path")

    output_config = None

    args = parser.parse_args()
    with open(args.train_config_file) as f:
        train_config = json.load(f)
        output_config = train_config

    train_df = get_dataset_df(train_config["data_dir"], train_config["sr"], train_config["n_mfcc"],
                              train_config["hop_length"],
                              train_config["n_fft"],
                              train_config["delta_width"])

    template_df = get_template_df(train_df, train_config["n_sample_per_word"])
    new_template_df = reduce_models(template_df, train_config["syllables_file_path"])

    if not args.test_config_file:
        choice_indexes = np.random.choice(len(train_df), 100, replace=False)
        test_mfcc_features = list(train_df.loc[choice_indexes, "mfcc_features"])
        reals = list(train_df.loc[choice_indexes, "label"])
    else:
        with open(args.test_config_file) as f:
            test_config = json.load(f)
            output_config = test_config
        test_df = get_dataset_df(test_config["data_dir"], test_config["sr"], test_config["n_mfcc"],
                                 test_config["hop_length"],
                                 test_config["n_fft"],
                                 test_config["delta_width"])

        test_mfcc_features = list(test_df["mfcc_features"])
        reals = test_df["label"]

    labels = ['sil', '1', 'tram', '4', 'muoi', '9', 'trieu', '3', '7', '8', 'nghin', '6', '5', 'lam', '2', 'tu',
              '0', 'mot', 'linh', 'm1']
    preds = predict(new_template_df, test_mfcc_features, 2)

    result_folder_path = "../output/result"
    helper.save_result(preds, reals, labels, result_folder_path, output_config)
