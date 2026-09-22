import numpy as np
import pandas as pd
import pytest

from noise_analyzer.io.loaders import load_signal_csv


def test_load_sample_only_csv(tmp_path):
    file_path = tmp_path / "signal.csv"

    dataframe = pd.DataFrame(
        {
            "signal": [
                0.124,
                0.482,
                -0.137,
                -0.521,
            ]
        }
    )

    dataframe.to_csv(file_path, index=False)

    record = load_signal_csv(
        file_path=file_path,
        sample_rate_hz=10000.0,
    )

    assert record.number_of_samples == 4
    assert np.isclose(record.sample_rate_hz, 10000.0)
    assert np.isclose(record.duration_seconds, 4 / 10000.0)
    assert np.isclose(record.nyquist_frequency_hz, 5000.0)

    assert np.allclose(
        record.samples,
        np.array([0.124, 0.482, -0.137, -0.521]),
    )

    assert record.channel_name == "signal"
    assert record.units == "unknown"
    assert record.source_file == file_path

def test_load_signal_csv_rejects_missing_signal_column(tmp_path):
    file_path = tmp_path / "wrong_column.csv"

    pd.DataFrame(
        {
            "measurement": [1.0, 2.0, 3.0]
        }
    ).to_csv(file_path, index=False)

    with pytest.raises(
        ValueError,
        match="Signal column 'signal' was not found",
    ):
        load_signal_csv(
            file_path=file_path,
            sample_rate_hz=1000.0,
        )

def test_load_signal_csv_rejects_nonnumeric_values(tmp_path):
    file_path = tmp_path / "nonnumeric.csv"

    pd.DataFrame(
        {
            "signal": [1.0, "bad", 3.0]
        }
    ).to_csv(file_path, index=False)

    with pytest.raises(
        ValueError,
        match="must contain numeric values",
    ):
        load_signal_csv(
            file_path=file_path,
            sample_rate_hz=1000.0,
        )

def test_load_signal_csv_rejects_missing_file(tmp_path):
    file_path = tmp_path / "does_not_exist.csv"

    with pytest.raises(FileNotFoundError):
        load_signal_csv(
            file_path=file_path,
            sample_rate_hz=1000.0,
        )

def test_load_time_signal_csv_infers_sample_rate(tmp_path):
    file_path = tmp_path / "time_signal.csv"

    dataframe = pd.DataFrame(
        {
            "time": [
                0.0000,
                0.0001,
                0.0002,
                0.0003,
            ],
            "signal": [
                0.124,
                0.482,
                -0.137,
                -0.521,
            ],
        }
    )

    dataframe.to_csv(file_path, index=False)

    record = load_signal_csv(file_path=file_path)

    assert record.number_of_samples == 4
    assert np.isclose(record.sample_rate_hz, 10000.0)
    assert np.isclose(record.sampling_interval_seconds, 0.0001)
    assert np.isclose(record.nyquist_frequency_hz, 5000.0)

def test_load_time_signal_csv_rejects_nonuniform_timestamps(tmp_path):
    file_path = tmp_path / "nonuniform.csv"

    dataframe = pd.DataFrame(
        {
            "time": [
                0.0000,
                0.0001,
                0.0002,
                0.00035,
                0.0004,
            ],
            "signal": [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
            ],
        }
    )

    dataframe.to_csv(file_path, index=False)

    with pytest.raises(
        ValueError,
        match="not uniformly sampled",
    ):
        load_signal_csv(file_path=file_path)

def test_sample_only_csv_requires_sample_rate(tmp_path):
    file_path = tmp_path / "signal_only.csv"

    pd.DataFrame(
        {
            "signal": [1.0, 2.0, 3.0]
        }
    ).to_csv(file_path, index=False)

    with pytest.raises(
        ValueError,
        match="sample_rate_hz must be provided",
    ):
        load_signal_csv(file_path=file_path)

def test_load_time_signal_csv_rejects_non_increasing_time(tmp_path):
    file_path = tmp_path / "bad_time.csv"

    pd.DataFrame(
        {
            "time": [
                0.0000,
                0.0001,
                0.0001,
                0.0002,
            ],
            "signal": [
                1.0,
                2.0,
                3.0,
                4.0,
            ],
        }
    ).to_csv(file_path, index=False)

    with pytest.raises(
        ValueError,
        match="strictly increasing",
    ):
        load_signal_csv(file_path=file_path)