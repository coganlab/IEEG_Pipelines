import numpy as np
from ieeg.arrays.label import LabeledArray


def test_nanmean_nanstd_override():
    arr = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]], dtype=np.float32)
    labels = (("r0", "r1"), ("c0", "c1", "c2"))
    la = LabeledArray(arr, labels)

    mean_axis1 = np.nanmean(la, axis=1)
    assert isinstance(mean_axis1, LabeledArray)
    assert np.allclose(mean_axis1, np.nanmean(arr, axis=1))
    assert list(mean_axis1.labels[0]) == list(labels[0])

    std_axis0 = np.nanstd(la, axis=0)
    assert isinstance(std_axis0, LabeledArray)
    assert np.allclose(std_axis0, np.nanstd(arr, axis=0))

    mean2, std2 = la.nanmean_std(axis=1)
    assert np.allclose(mean2, np.nanmean(arr, axis=1))
    assert np.allclose(std2, np.nanstd(arr, axis=1))

    std_ddof = np.nanstd(la, axis=1, ddof=1)
    assert np.allclose(std_ddof, np.nanstd(arr, axis=1, ddof=1))

    mean_all = np.nanmean(la, axis=(0, 1))
    assert np.isclose(mean_all, np.nanmean(arr))
