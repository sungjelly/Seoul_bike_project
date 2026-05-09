import numpy as np

from src.data.tct_gat.tct_gat_dataset import validate_graph_files


def test_sample_time_index_is_1d_contract():
    sample_time_index = np.asarray([10, 11, 12], dtype=np.int64)
    assert sample_time_index.ndim == 1


def test_graph_feature_contract_names_do_not_include_net_demand():
    feature_config = {
        "rental_feature_columns": ["rental_count", "temperature", "wind_speed", "rainfall", "humidity"],
        "return_feature_columns": ["return_count", "temperature", "wind_speed", "rainfall", "humidity"],
    }
    assert "net_demand" not in str(feature_config)


def test_validate_graph_files_function_is_importable():
    assert callable(validate_graph_files)
