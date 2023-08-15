def _qt_backend():
    """Set the backend to Qt5Agg"""
    import matplotlib
    if matplotlib.get_backend() not in ['Qt5Agg', 'headless']:
        matplotlib.use('Qt5Agg')


_qt_backend()
