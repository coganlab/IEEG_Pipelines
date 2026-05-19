import time
import numpy as np
from ieeg.arrays.label import LabeledArray


def bench(name, fn, n=5, warm=1):
    for _ in range(warm):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    dt = (time.perf_counter() - t0) / n
    print(f"{name:45s} {dt:.6f}s")
    return dt


def make_labels(prefix, n):
    return tuple(f"{prefix}{i}" for i in range(n))


if __name__ == "__main__":
    np.random.seed(0)

    size = 2000
    labs = (make_labels("r", size), make_labels("c", size))
    a = LabeledArray(np.random.randn(size, size).astype(np.float32), labs)

    size_small = 256
    labs_s = (make_labels("rs", size_small), make_labels("cs", size_small))
    a_small = LabeledArray(np.random.randn(size_small, size_small).astype(np.float32), labs_s)

    shape3 = (320, 240, 160)
    labs3 = (make_labels("x", shape3[0]), make_labels("y", shape3[1]), make_labels("z", shape3[2]))
    a3 = LabeledArray(np.random.randn(*shape3).astype(np.float32), labs3)

    mask = np.zeros(size, dtype=bool)
    mask[::3] = True
    mask_small = np.zeros(size_small, dtype=bool)
    mask_small[::5] = True

    idx_int_1d = np.arange(0, size, 7, dtype=np.intp)
    idx_lbl_1d = np.array([f"c{i}" for i in idx_int_1d])
    idx_int_1d_small = np.arange(0, size_small, 5, dtype=np.intp)
    idx_lbl_1d_small = np.array([f"cs{i}" for i in idx_int_1d_small])

    idx_int_2d = (np.arange(12, dtype=np.intp).reshape(3, 4)) % size_small
    idx_lbl_2d = np.array([[f"cs{v}" for v in row] for row in idx_int_2d])

    idx_int_3d = (np.arange(24, dtype=np.intp).reshape(2, 3, 4)) % shape3[2]
    idx_lbl_3d = np.array([[[f"z{v}" for v in row] for row in plane] for plane in idx_int_3d])

    taa_idx = (np.arange(shape3[0] * shape3[1] * 8, dtype=np.intp) % shape3[2]).reshape(shape3[0], shape3[1], 8)

    print("== indexing ==")
    bench("scalar label", lambda: a["r10", "c20"])
    bench("scalar int", lambda: a[10, 20])
    bench("row slice", lambda: a[10, :])
    bench("col slice", lambda: a[:, 10])
    bench("mask axis1", lambda: a[:, mask])
    bench("list labels axis1", lambda: a[:, idx_lbl_1d])
    bench("list ints axis1", lambda: a[:, idx_int_1d])
    bench("2d ints axis1", lambda: a_small[:, idx_int_2d])
    bench("2d labels axis1", lambda: a_small[:, idx_lbl_2d])
    bench("ix_ ints", lambda: a[np.ix_(idx_int_1d_small, idx_int_1d_small)])

    print("== take ==")
    bench("np.take ints 1d", lambda: np.take(a, idx_int_1d, axis=1))
    bench("np.take labels 1d", lambda: np.take(a, idx_lbl_1d, axis=1))
    bench("np.take ints 2d", lambda: np.take(a_small, idx_int_2d, axis=1))
    bench("np.take labels 2d", lambda: np.take(a_small, idx_lbl_2d, axis=1))
    bench("np.take ints 3d", lambda: np.take(a3, idx_int_3d, axis=2))
    bench("np.take labels 3d", lambda: np.take(a3, idx_lbl_3d, axis=2))
    bench("LabeledArray.take 1d", lambda: a.take(idx_int_1d, axis=1))
    bench("LabeledArray.take labels", lambda: a.take(idx_lbl_1d, axis=1))
    bench("LabeledArray.take 3d", lambda: a3.take(idx_int_3d, axis=2))
    bench("take_along_axis", lambda: np.take_along_axis(a3, taa_idx, axis=2))

    print("== shape ops ==")
    bench("transpose", lambda: a.transpose())
    bench("swapaxes", lambda: a.swapaxes(0, 1))
    bench("reshape", lambda: a_small.reshape((size_small * size_small, )))
    bench("stack", lambda: np.stack((a_small, a_small), axis=0))
    bench("concat axis0", lambda: np.concatenate((a_small[:128], a_small[128:]), axis=0))
    bench("concat axis1", lambda: np.concatenate((a_small[:, :128], a_small[:, 128:]), axis=1))
    bench("combine", lambda: a_small.combine((0, 1)))

    print("== misc ==")
    a_nan = a_small.copy()
    a_nan[:10, :10] = np.nan
    bench("dropna", lambda: a_nan.dropna())
    bench("ufunc add", lambda: a + 1)
    bench("labels property", lambda: a.labels)
    bench("find label", lambda: a.find("c10", axis=1))

    # SentenceRep-style patterns
    print("== SentenceRep patterns ==")
    conds = ("aud_ls", "aud_lm", "aud_jl", "go_ls", "go_lm", "resp")
    epochs = ("pre", "post")
    trials = tuple(f"t{i}" for i in range(30))
    channels = tuple(f"D{d:04d}-X{c:02d}" for d in range(1, 20) for c in range(160))
    times = tuple(str(i) for i in range(200))

    arr5 = LabeledArray(
        np.random.randn(len(conds), len(epochs), len(trials),
                        len(channels), len(times)).astype(np.float32),
        labels=(conds, epochs, trials, channels, times),
    )

    cond_pair = ["aud_ls", "aud_lm"]
    ch_idx = list(range(0, len(channels), 3))
    ch_mask = np.zeros(len(channels), dtype=bool)
    ch_mask[ch_idx] = True

    bench("conds list index", lambda: arr5[cond_pair,])
    bench("conds then chan idx", lambda: arr5[cond_pair,][:, :, :, ch_idx])
    bench("chan idx then conds", lambda: arr5[:, :, :, ch_idx][cond_pair,])
    bench("np.take cond scalar", lambda: np.take(arr5, "aud_ls", axis=0))
    bench("np.take cond list", lambda: np.take(arr5, np.array(cond_pair), axis=0))
    bench("boolean mask channels", lambda: arr5[:, :, :, ch_mask])
    bench("concat label slices", lambda: np.concatenate([arr5[c] for c in cond_pair], axis=-1))
    bench("hstack label slices", lambda: np.hstack([arr5[c] for c in cond_pair]))
    bench("combine trial+chan", lambda: arr5.combine((2, 3)))
    bench("dropna on 5d", lambda: arr5.dropna())
    bench("ravel", lambda: arr5.ravel())

    print("== nan reductions (large) ==")
    shape6 = (7, 4, 512, 31, 4, 100)
    labs6 = (
        make_labels("a", shape6[0]),
        make_labels("b", shape6[1]),
        make_labels("c", shape6[2]),
        make_labels("d", shape6[3]),
        make_labels("e", shape6[4]),
        make_labels("f", shape6[5]),
    )
    print("Creating array with shape", shape6)
    arr6 = LabeledArray(np.random.randn(*shape6).astype(np.float32), labs6)
    print("Introducing NaNs")
    arr6[..., :20] = np.nan
    bench("np.nanmean labeled axis=-1", lambda: np.nanmean(arr6, axis=-1), n=3)
    bench("np.nanmean base axis=-1", lambda: np.nanmean(np.asarray(arr6), axis=-1), n=3)
    bench("np.nanstd labeled axis=-1", lambda: np.nanstd(arr6, axis=-1), n=3)
    bench("nanmean_std labeled axis=-1", lambda: arr6.nanmean_std(axis=-1), n=3)
    bench("np.nanmean labeled axis=(-2,-1)", lambda: np.nanmean(arr6, axis=(-2, -1)), n=3)
    bench("np.nanstd labeled axis=(-2,-1)", lambda: np.nanstd(arr6, axis=(-2, -1)), n=3)

