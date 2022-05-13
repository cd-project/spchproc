Install dependencies:

- pip install -r requirements.txt


DTW:
// Change cfgs in dtw.json to match files
- python3 dtw_recg.py -conf dtw.json

HMM:
// Create train data for HMM
-python3 mfcc.py --hmm_train_create hmm_create_train.json --hmm_test_create hmm_create_test.json
-python3 hmm.py --train_cfg_path hmm_train_config.json --length_cfg_path hmm_test_config.json
