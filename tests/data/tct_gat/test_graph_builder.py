import numpy as np

from src.data.tct_gat.graph_builder import (
    CorrelationMatrices,
    ODFeatureMatrices,
    build_edge_attributes,
    correlation_matrix,
    haversine_distance_matrix,
    lagged_cross_correlation,
    operation_pair_one_hot,
    top_k_neighbors,
)


def test_operation_pair_one_hot_handles_directed_pairs():
    one_hot, unknown = operation_pair_one_hot(
        np.asarray(["LCD", "QR", "LCD", "QR", ""]),
        np.asarray(["QR", "LCD", "LCD", "QR", "LCD"]),
    )

    expected = np.asarray(
        [
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(one_hot, expected)
    np.testing.assert_array_equal(unknown, np.asarray([False, False, False, False, True]))


def test_top_k_neighbor_index_is_target_major():
    scores = np.asarray(
        [
            [0.0, 0.9, 0.2],
            [0.7, 0.0, 0.3],
            [0.4, 0.1, 0.0],
        ],
        dtype=np.float32,
    )
    fallback = np.ones_like(scores)

    neighbor_index, _ = top_k_neighbors(scores, 2, fallback)

    assert neighbor_index.shape == (3, 2)
    assert neighbor_index[0, 0] == 1
    assert neighbor_index[1, 0] == 0
    assert neighbor_index[2, 0] == 1


def test_lagged_cross_correlation_finds_expected_best_lag():
    source = np.asarray([[1.0], [3.0], [2.0], [5.0], [4.0], [8.0], [7.0]], dtype=np.float32)
    target = np.asarray([[0.0], [0.0], [1.0], [3.0], [2.0], [5.0], [4.0]], dtype=np.float32)

    corr, best_lag = lagged_cross_correlation(source, target, [1, 2, 3])

    assert corr.shape == (1, 1)
    assert best_lag[0, 0] == 2
    assert np.isclose(corr[0, 0], 1.0)


def test_edge_attr_contains_no_nan_or_inf():
    lat = np.asarray([37.5, 37.51, 37.52], dtype=np.float32)
    lon = np.asarray([127.0, 127.01, 127.02], dtype=np.float32)
    dist = haversine_distance_matrix(lat, lon)
    inv = (1.0 / (1.0 + dist)).astype(np.float32)
    same_district = np.eye(3, dtype=np.float32)
    flow = np.asarray(
        [
            [1, 2, 0],
            [0, 1, 3],
            [4, 0, 1],
        ],
        dtype=np.int32,
    )
    float_flow = flow.astype(np.float32)
    od = ODFeatureMatrices(
        od_flow=flow,
        log_od_flow=np.log1p(float_flow),
        od_probability=np.zeros((3, 3), dtype=np.float32),
        reverse_od_probability=np.zeros((3, 3), dtype=np.float32),
        mean_duration_min=float_flow * 10.0,
        mean_trip_distance_km=float_flow,
        mean_duration_lag_bins=float_flow,
    )
    corr = correlation_matrix(np.asarray([[1, 2, 3], [2, 1, 3], [3, 4, 3]], dtype=np.float32))
    correlations = CorrelationMatrices(
        rental_corr=corr,
        return_corr=corr,
        rental_to_return_corr=corr,
        best_rental_to_return_lag=np.ones((3, 3), dtype=np.float32),
        return_to_rental_corr=corr,
        best_return_to_rental_lag=np.ones((3, 3), dtype=np.float32) * 2,
    )
    sources = np.asarray([[0, 1], [1, 2], [2, 0]], dtype=np.int64)
    targets = np.asarray([[0, 0], [1, 1], [2, 2]], dtype=np.int64)

    edge_attr, _ = build_edge_attributes(
        sources,
        targets,
        dist,
        inv,
        same_district,
        np.asarray(["LCD", "QR", "LCD"], dtype=object),
        od,
        correlations,
        max_lag=4,
    )

    assert np.isfinite(edge_attr).all()


def test_rr_dd_self_edge_inserted_at_k0():
    scores = np.asarray(
        [
            [0.1, 0.9, 0.2],
            [0.7, 0.1, 0.3],
            [0.4, 0.1, 0.1],
        ],
        dtype=np.float32,
    )
    fallback = np.ones_like(scores)

    neighbor_index, _ = top_k_neighbors(scores, 2, fallback, force_self_first=True)

    np.testing.assert_array_equal(neighbor_index[:, 0], np.arange(3))
