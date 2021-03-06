import copy

from mfcc import *
from dtw import dtw


def predict(template_df: pd.DataFrame, test_mfcc_features: list, warp):
    preds = []
    for mfcc_test_i in test_mfcc_features:
        dt_list_i = []

        for j in range(len(template_df)):
            mfccj = template_df.iloc[j]["mfcc_features"]
            dist_list = [lambda x, y: np.linalg.norm(x - y, ord=2)] * len(mfccj)

            dtj, _, _ = dtw(mfccj, mfcc_test_i, dist_list, warp=warp)
            dt_list_i.append(dtj)

        preds.append(template_df.iloc[np.argmin(dt_list_i)]["label"])

    return preds


def k_means(mfcc_features: list, n_state: int):

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


def minimize_mfcc(template_df: pd.DataFrame, labelconf_path: str):
    syllable_df = pd.read_csv(labelconf_path, sep="\t", header=None)
    print(syllable_df.columns)
    syllable_df.rename(columns={0: "label", 1: "n_syllables"}, inplace=True)
    
    new_df = pd.DataFrame(columns=list(template_df.columns) + ["conv"])
    for label in template_df["label"].unique(): 
        mfcc_features = list(template_df[template_df["label"] == label]["mfcc_features"])
        # n_state = syllable_df["n_syllables"][syllable_df["label"] == label].iloc[0]

        reduced_mfcc_feature, new_cov = k_means(mfcc_features, 1)
        new_df.loc[len(new_df)] = [label, reduced_mfcc_feature, new_cov]

    return new_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-conf")

    output_config = None

    args = parser.parse_args()
    with open(args.train_config_file) as f:
        train_config = json.load(f)
        output_config = train_config

    train_df = get_dataset_df(train_config["data_dir"], train_config["sr"], train_config["n_mfcc"],
                              train_config["hop_length"],
                              train_config["n_fft"],
                              train_config["delta_width"])

    template_df = get_template_df(train_df, train_config["wsa_num"])
    new_template_df = minimize_mfcc(template_df, train_config["labelconf_path"])

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

    labels = ['sil','trai', 'phai', 'len', 'xuong', 'nhay', 'ban', 'A', 'B']
    preds = predict(new_template_df, test_mfcc_features, 2)

    result_folder_path = "../output/result"
    utils.save_result(preds, reals, labels, result_folder_path, output_config)
