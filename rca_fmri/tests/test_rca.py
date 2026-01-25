"""Tests for the RCA transformer."""
import numpy as np
import pytest
from rca_fmri import RCA


@pytest.fixture
def data():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(6, 4)).astype("float32")
    labels = np.array(
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [2, 0],
            [2, 1],
        ],
        dtype=int,
    )
    return X, labels


def test_rca_transformer(data):
    """Check the internals and behaviour of `RCA`."""
    X, labels = data
    trans = RCA(
        n_components=1,
        n_epochs=1,
        batch_size=2,
        model_type="linear",
    )

    trans.fit(X, labels)
    assert hasattr(trans, "weights_")

    X_trans = trans.transform(X)
    assert X_trans.shape == (X.shape[0], 1)
