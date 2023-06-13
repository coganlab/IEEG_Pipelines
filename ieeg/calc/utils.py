import numpy as np


def get_elbow(data: np.ndarray) -> int:
    """Draws a line between the first and last points in a dataset and finds
    the point furthest from that line.

    Parameters
    ----------
    data : array
        The data to find the elbow in.

    Returns
    -------
    int
        The index of the elbow point.
    """
    nPoints = len(data)
    allCoord = np.vstack((range(nPoints), data)).T
    np.array([range(nPoints), data])
    firstPoint = allCoord[0]
    lineVec = allCoord[-1] - allCoord[0]
    lineVecNorm = lineVec / np.sqrt(np.sum(lineVec ** 2))
    vecFromFirst = allCoord - firstPoint
    scalarProduct = np.sum(vecFromFirst * np.matlib.repmat(
        lineVecNorm, nPoints, 1), axis=1)
    vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
    vecToLine = vecFromFirst - vecFromFirstParallel
    distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))
    # set distance to points below lineVec to 0
    distToLine[vecToLine[:, 1] < 0] = 0
    idxOfBestPoint = np.argmax(distToLine)
    return idxOfBestPoint


def stitch_mats(mats: list[np.ndarray], overlaps: list[int], axis: int = 0
                ) -> np.ndarray:
    """break up the matrices into their overlapping and non-overlapping parts
    then stitch them back together

    Parameters
    ----------
    mats : list
        The matrices to stitch together
    overlaps : list
        The number of overlapping rows between each matrix
    axis : int, optional
        The axis to stitch along, by default 0

    Returns
    -------
    np.ndarray
        The stitched matrix
    """
    stitches = [mats[0]]
    if len(mats) != len(overlaps) + 1:
        raise ValueError("The number of matrices must be one more than the num"
                         "ber of overlaps")
    for i, over in enumerate(overlaps):
        stitches = stitches[:-2] + merge(stitches[-1], mats[i+1], over, axis)
    return np.concatenate(stitches, axis=axis)


def merge(mat1: np.ndarray, mat2: np.ndarray, overlap: int, axis: int = 0
          ) -> list[np.ndarray]:
    """Take two arrays and merge them over the overlap gradually"""
    sl = [slice(None)] * mat1.ndim
    sl[axis] = slice(0, mat1.shape[axis]-overlap)
    start = mat1[tuple(sl)]
    sl[axis] = slice(mat1.shape[axis]-overlap, mat1.shape[axis])
    middle1 = np.multiply(np.linspace(1, 0, overlap), mat1[tuple(sl)])
    sl[axis] = slice(0, overlap)
    middle2 = np.multiply(np.linspace(0, 1, overlap), mat2[tuple(sl)])
    middle = np.add(middle1, middle2)
    sl[axis] = slice(overlap, mat2.shape[axis])
    last = mat2[tuple(sl)]
    return [start, middle, last]
