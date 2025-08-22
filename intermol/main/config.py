import time

# DATA_CONFIG
DATA_PATH = '../../0_0025/sae_mols_ss0_0025_no_doubles_max200_2739074.parquet'
COL_SELE = 'c_smiles'
BATCH_SIZE = 128
TRAIN_SIZE = 0.8

# SAE_CONFIG
BASE_HOOK_POS = 9
EXP_F = 4
K = 128
DEAD_STEPS_THRESH = 5000
WEIGHTS_PATH = None

# RUN_CONFIG
EPOCHS = 1
LR = 2e-4
WD = 1e-2
SEED = int(time.time())