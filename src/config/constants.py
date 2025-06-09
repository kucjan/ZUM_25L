# Experiments
SEED = 42
DATASET_IDS = [
    350,  # Default of credit cards clients
    891,  # CDC Diabetes Health Indicators
    697,  # Predict Students' Dropout and Academic Success
    848,  # Secondary Mushroom
    186,  # Wine quality
    17,  # Breast Cancer Wisconsin
]
NUM_OF_RUNS = 10
TEST_SPLIT_SIZE = 0.3

# OneClassWrapper modifiable params
OUTLIER_RATIO = 0.5
EXTEND_FACTOR = 0.2
EXTREME_FACTOR = 0.1
EXTREME_SCALER = 2

# OneClassWrapper constants
ADD_CATEGORIES_NUM = 3
MIN_UNIQUE_NUM_VALUES = 30
