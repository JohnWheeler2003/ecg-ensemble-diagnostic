import pandas as pd
import numpy as np
import wfdb
import ast
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

from src.config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, SAMPLE_RATE, TARGET_CLASSES, FILTER_LOWCUT, FILTER_HIGHCUT, FILTER_ORDER, TRAIN_FOLDS, VAL_FOLD, TEST_FOLD
)

def load_metadata():
    """Loads and parses the PTB-XL metadata and SCP mappings"""
    print("Loading metadata...")
    Y = pd.read_csv(RAW_DATA_DIR / 'ptbxl_database.csv', index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    agg_df = pd.read_csv(RAW_DATA_DIR / 'scp_statements.csv', index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    return Y, agg_df

def aggregate_diagnostic(y_dic, agg_df):
    """Maps raw SCP codes to the 5 Superclasses."""
    tmp = []
    for key in y_dic.keys():
        if key in agg_df.index:
            diag_class = agg_df.loc[key].diagnostic_class
            if diag_class in TARGET_CLASSES:
                tmp.append(diag_class)
    return list(set(tmp))

def generate_labels(Y, agg_df):
    """Creates a multi-hot encoded binary matrix for the labels."""
    print("Mapping labels to Superclasses...")
    Y['diagnostic_superclass'] = Y.scp_codes.apply(lambda x: aggregate_diagnostic(x, agg_df))

    # Filter out records that don't belong to any of the 5 target classes
    Y = Y[Y['diagnostic_superclass'].map(len) > 0]

    mlb = MultiLabelBinarizer(classes=TARGET_CLASSES)
    multi_hot_labels = mlb.fit_transform(Y['diagnostic_superclass'])

    return Y, multi_hot_labels

def butter_bandpass_filter(data, lowcut, highcut, fs, order):
    """Applies Butterworth Bandpass filter"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    # filtfilt applies the filter forward and backward to ensure zero phase shift 
    y = filtfilt(b, a, data, axis=0)
    return y

def load_and_process_signals(df):
    """Loads .dat files, applies bandpass filter, and Z-score normalization."""
    print(f"Processing {len(df)} ECG signals...")
    processed_signals = []

    for filename in tqdm(df.filename_lr):
        filepath = str(RAW_DATA_DIR / filename)
        signal, meta = wfdb.rdsamp(filepath)

        # Signal Cleaning (Butterworth Bandpass)
        filtered_signal = butter_bandpass_filter(
            signal, FILTER_LOWCUT, FILTER_HIGHCUT, SAMPLE_RATE, FILTER_ORDER
        )

        # Lead-wise Z-score Normalization
        # Mean and STD calculated per lead (axis 0 is time, axis 1 is lead)
        mean = np.mean(filtered_signal, axis=0)
        std = np.std(filtered_signal, axis=0)
        # Add small epsilon to prevent division by zero
        normalized_signal = (filtered_signal - mean) / (std + 1e-8)

        processed_signals.append(normalized_signal)
    return np.array(processed_signals)

def main():
    # 1. Ingestion & Label Mapping
    Y_raw , agg_df = load_metadata()
    Y, multi_hot_labels = generate_labels(Y_raw, agg_df)

    # 2. Signal Processing and Normalization
    X = load_and_process_signals(Y)

    print(f"Final Signal Matrix Shape: {X.shape}") # Expected (N, 1000, 12)
    print(f"Final Label Matrix: {multi_hot_labels.shape}") # Expected (N, 5)

    # 3. Stratification
    print("Splitting data into Train, Validation and Test sets...")
    train_mask = Y.strat_fold.isin(TRAIN_FOLDS)
    val_mask = Y.strat_fold == VAL_FOLD
    test_mask = Y.strat_fold == TEST_FOLD

    X_train, y_train = X[train_mask], multi_hot_labels[train_mask]
    X_val, y_val = X[val_mask], multi_hot_labels[val_mask]
    X_test, y_test = X[test_mask], multi_hot_labels[test_mask]

    # 4. Save to Disk
    print("Saving processed data...")
    np.save(PROCESSED_DATA_DIR / 'X_train.npy', X_train)
    np.save(PROCESSED_DATA_DIR / 'y_train.npy', y_train)
    np.save(PROCESSED_DATA_DIR / "X_val.npy", X_val)
    np.save(PROCESSED_DATA_DIR / 'y_val.npy', y_val)
    np.save(PROCESSED_DATA_DIR / 'X_test.npy', X_test)
    np.save(PROCESSED_DATA_DIR / 'y_test.npy', y_test)

    print("Phase 1 Data Engineering Complete")

if __name__ == "__main__":
    main()