import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from pathlib import Path
from src.data_prep import (
    load_metadata,
    aggregate_diagnostic,
    generate_labels,
    butter_bandpass_filter,
    load_and_process_signals,
)

from src.config import TARGET_CLASSES, SAMPLE_RATE


# 1. Test for load_metadata
@patch("src.data_prep.pd.read_csv")
def test_load_metadata_standard(mock_read_csv):
    """Standard Input: Verifies CSVs are read and scp_codes are parsed into dicts"""
    # Create dummy data for two CSV calls
    # Note load_metadata calls read_csv twice. side_effect handles multiple calls
    mock_db = pd.DataFrame(
        {
            "ecg_id": [1],
            "scp_codes": ["{'NORM': 100}"],  # String format as it would appear in a CSV
        }
    ).set_index("ecg_id")

    mock_scp = pd.DataFrame(
        {"diagnostic": [1, 0], "diagnostic_class": ["NORM", "IGNORE"]},
        index=["NORM", "OTHER"],
    )

    mock_read_csv.side_effect = [mock_db, mock_scp]

    # Act
    Y, agg_df = load_metadata()

    # Assert
    # Check if scp_codes was successfully converted from string to dict
    assert isinstance(Y.iloc[0]["scp_codes"], dict)
    assert Y.iloc[0]["scp_codes"]["NORM"] == 100

    # Check if agg_df was filtered (Only 'diagnostic == 1' should remain)
    assert len(agg_df) == 1
    assert "OTHER" not in agg_df.index


@patch("src.data_prep.pd.read_csv")
def test_load_metadata_malformed_dict(mock_read_csv):
    """Erroneous Input: What if scp_codes column has corrupted string data?"""
    # Create dataframe with a string that ast.literal_eval cannot parse
    mock_db = pd.DataFrame(
        {"ecg_id": [1], "scp_codes": ["{'NORM': 100, malformed_part}"]}
    ).set_index("ecg_id")

    mock_read_csv.side_effect = [mock_db, MagicMock()]

    # Expect SyntaxError or ValueError from ast.literal_eval
    with pytest.raises((SyntaxError, ValueError)):
        load_metadata()


def test_load_metadata_file_not_found():
    """Edge Case: script is run, but data/raw/ directory is empty"""
    # Nothing to mock, just need to point at non-existent path and verify FileNotFoundError
    mock_path = Path("/non/existent/path")
    with patch("src.data_prep.RAW_DATA_DIR", new=mock_path):
        with pytest.raises(FileNotFoundError):
            load_metadata()


# 2. Test for aggregate_diagnostic
@pytest.fixture
def mock_agg_df():
    """Fixture providing a standard mapping dataframe"""
    return pd.DataFrame(
        {"diagnostic_class": ["STTC", "NORM", "MI", "IGNORE"]},
        index=["NDT", "NORM", "AMI", "PAC"],
    )


def test_aggregate_diagnostic_standard(mock_agg_df):
    """Standard Input: Valid dictionary with known SCP codes"""
    y_dic = {"NDT": 100, "NORM": 100}
    result = aggregate_diagnostic(y_dic, mock_agg_df)
    assert set(result) == {"STTC", "NORM"}


def test_aggregate_diagnostic_edge_non_target(mock_agg_df):
    """Edge Case: Contains codes that map to non-target classes or don't exist"""
    y_dic = {"PAC": 100, "UNKNOWN_CODE": 50}
    result = aggregate_diagnostic(y_dic, mock_agg_df)
    assert result == [], "Should return empty list for non-target or unknown codes"


def test_aggregate_diagnostic_empty_input(mock_agg_df):
    """Edge Case: Empty dictionary"""
    y_dic = {}
    result = aggregate_diagnostic(y_dic, mock_agg_df)
    assert result == [], "Empty input should yield empty output"


# 3. Test for generate_labels
@pytest.fixture
def mock_Y_df():
    """Fixture providing a mock patient dataframe"""
    return pd.DataFrame(
        {
            "ecg_id": [1, 2, 3],
            "scp_codes": [
                {"NDT": 100},  # Maps to STTC
                {"NORM": 100, "AMI": 100},  # Maps to NORM, MI
                {"PAC": 100},  # Will map to nothing (dropped)
            ],
        }
    ).set_index("ecg_id")


def test_generate_labels_standard_and_dropping(mock_Y_df, mock_agg_df):
    """Standard & Edge: Verifies correct multi-hot encoding and dropping of empty rows"""
    Y_processed, multi_hot = generate_labels(mock_Y_df, mock_agg_df)
    assert len(Y_processed) == 2, "Failed to drop rows without target superclass"
    assert multi_hot.shape == (2, len(TARGET_CLASSES))

    # Verify exact multi-hot encoding for Patient 2 (NORM and MI)
    expected_patient_2 = np.array([1, 1, 0, 0, 0])
    np.testing.assert_array_equal(multi_hot[1], expected_patient_2)


def test_generate_labels_error_missing_column():
    """Erroneous Input: Dataframe missing the 'scp_codes' column"""
    bad_df = pd.DataFrame({"wrong_column": [1, 2, 3]})
    mock_agg = pd.DataFrame()
    with pytest.raises(AttributeError):
        generate_labels(bad_df, mock_agg)


# 4. Tests for butter_bandpass_filter
def test_butter_bandpass_filter_standard():
    """Standard Input: Applies filter to random noise"""
    np.random.seed(42)
    fake_signal = np.random.randn(1000, 12)
    filtered = butter_bandpass_filter(fake_signal, 0.5, 40.0, SAMPLE_RATE, 4)

    assert filtered.shape == (1000, 12)
    assert not np.array_equal(fake_signal, filtered), "Filter did not modify the signal"


def test_butter_bandpass_filter_edge_zeroes():
    """Edge Case: Filtering a completely flat signal (all zeroes)"""
    flat_signal = np.zeros((1000, 12))
    filtered = butter_bandpass_filter(flat_signal, 0.5, 40.0, SAMPLE_RATE, 4)
    np.testing.assert_almost_equal(filtered, flat_signal)


def test_butter_bandpass_filter_error_nyquist():
    """Erroneous Input: Highcut frequency exceeds Nyquist limit (fs/2)"""
    fake_signal = np.random.randn(1000, 12)
    with pytest.raises(ValueError, match="Digital filter critical frequencies must be"):
        # 60 Hz highcut is invalid for a 100 Hz sampling rate (Nyquist limit is 50 Hz)
        butter_bandpass_filter(fake_signal, 0.5, 60.0, SAMPLE_RATE, 4)


# 5. Test for load_and_process_signals
@patch("src.data_prep.wfdb.rdsamp")
def test_load_and_process_signals_standard(mock_rdsamp):
    """Standard Input: Mocks wfdb to test processing and Z-score normalization"""
    # Create fake dataframe and fake waveform
    mock_df = pd.DataFrame({"filename_lr": ["patient_1"]})

    # Create dummy signal (1000 timesteps, 12 leads) with non-zero variance
    np.random.seed(42)
    dummy_waveform = np.random.randn(1000, 12) * 5 + 2  # Mean ~2, Std ~5
    mock_rdsamp.return_value = (dummy_waveform, {"meta": "data"})

    processed_X = load_and_process_signals(mock_df)

    assert processed_X.shape == (1, 1000, 12)

    # Check Z-score normalization (Mean should be ~0, Std should be ~1 across time axis)
    lead_0 = processed_X[0, :, 0]
    assert np.isclose(np.mean(lead_0), 0, atol=1e-6)
    assert np.isclose(np.std(lead_0), 1, atol=1e-6)


@patch("src.data_prep.wfdb.rdsamp")
def test_load_and_process_signals_edge_dead_lead(mock_rdsamp):
    """ "Edge Case: Tests the 1e-8 division by zero protection on a flatlining lead"""
    mock_df = pd.DataFrame({"filename_lr": ["patient_1"]})

    # Create signal where Lead 0 is completely flat (0 variance)
    dummy_waveform = np.random.randn(1000, 12)
    dummy_waveform[:, 0] = 0.0
    mock_rdsamp.return_value = (dummy_waveform, {"meta": "data"})

    # If the + 1e-8 epsilon fails, this will throw RuntimeWarning/ZeroDivision Error or result in NaNs
    processed_X = load_and_process_signals(mock_df)

    assert not np.isnan(processed_X).any(), "NaN values found due to division by zero!"


# 6. integration Test (Smoke Test)
@patch("src.data_prep.wfdb.rdsamp")
@patch("src.data_prep.pd.read_csv")
def test_main_smoke_integration(mock_read_csv, mock_rdsamp, tmp_path):
    """Smoke Test: Runs entire main() orchestration on a 3-patient dummy dataset"""
    # Mock Metadata (CSV reads)
    # Must include all columns that main() and its helper functions rely on, specifically 'strat_fold' and 'filename_lr'
    mock_db = pd.DataFrame(
        {
            "ecg_id": [1, 2, 3],
            "scp_codes": ["{'NORM': 100}", "{'AMI': 100}", "{'NDT': 100}"],
            "strat_fold": [1, 9, 10],
            "filename_lr": ["pat1", "pat2", "pat3"],
        }
    ).set_index("ecg_id")

    mock_scp = pd.DataFrame(
        {"diagnostic": [1, 1, 1], "diagnostic_class": ["NORM", "MI", "STTC"]},
        index=["NORM", "AMI", "NDT"],
    )

    mock_read_csv.side_effect = [mock_db, mock_scp]

    # Mock Waveforms (wfdb reads)
    np.random.seed(42)
    dummy_waveform = np.random.randn(1000, 12) * 2
    mock_rdsamp.return_value = (dummy_waveform, {"meta": "data"})

    # Intercept Disk Writes
    # Temporarily replace PROCESSED_DATA_DIR with tmp_path fixture to make sure np.save() writes test files to disposable folder
    with patch("src.data_prep.PROCESSED_DATA_DIR", new=tmp_path):
        from src.data_prep import main

        # Run entire pipeline
        main()

        # Verify orchestration successfully split and saved the data
        assert (tmp_path / "X_train.npy").exists()
        assert (tmp_path / "y_train.npy").exists()
        assert (tmp_path / "X_val.npy").exists()
        assert (tmp_path / "X_test.npy").exists()
