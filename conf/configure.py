class Config:
    SR = 32000
    RESAMPLE_SR = 16000
    EMB_LEN = 80000
    # Dataset
    ROOT_FOLDER = './data'
    # Training
    N_CLASSES = 2
    BATCH_SIZE = 192
    N_EPOCHS = 1
    N_FOLD = 5
    #LR = 3e-4
    # Others
    SEED = 78
    PRE_MODEL = "./data/wav2vec_pretrain"
    conf_dic = {
    "loss": "CCE",
    "model_config": {
        "nb_samp": 96900,
        "first_conv": 128,
        "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
        "gat_dims": [64, 32],
        "pool_ratios": [0.5, 0.7, 0.5, 0.5],
        "temperatures": [2.0, 2.0, 100.0, 100.0]
        }
    }