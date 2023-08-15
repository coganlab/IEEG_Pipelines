from mne.utils import logger


def _qt_backend():
    """Set the backend to Qt5Agg"""
    import matplotlib
    try:
        matplotlib.use('Qt5Agg')
    except ImportError:
        logger.warn("Qt5Agg backend not available, using default backend")


_qt_backend()
