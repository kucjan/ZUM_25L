# Experiments
SEED = 42
DATASET_IDS = [
    350,  # Default of credit cards clients
    365,  # Polish Companies Bankruptcy
    697,  # Predict Students' Dropout and Academic Success
    159,  # MAGIC Gamma Telescope
    186,  # Wine quality
    17,  # Breast Cancer Wisconsin
]
NUM_OF_RUNS = 10
TEST_SPLIT_SIZE = 0.3

# OneClassWrapper modifiable params
OUTLIER_RATIO = 0.4
GENERATOR_PARAMS = {
    "default": {
        "extend_factor": 0.2,
        "shrink_factor": 0.1,
    },
    "mahalanobis": {
        "extend_factor": 4.0,
        "noise_factor": 4.0,
        "extreme_ratio": 0.5,
        "extreme_extend_scaler": 2.0,
        "extreme_noise_scaler": 2.0,
    },
}

# OneClassWrapper constants
ADD_CATEGORIES_NUM = 3
MIN_UNIQUE_NUM_VALUES = 30
OUTLIER_DIRECTION_PROB = 0.5
