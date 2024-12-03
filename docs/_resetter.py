import time
import gc
import mne
import os
from mne.utils import (
    _assert_no_instances, # noqa, analysis:ignore
    sizeof_fmt,
)
from mne.viz import Brain # noqa, needed for mne.viz._brain._BrainScraper


class Resetter(object):
    """Simple class to make the str(obj) static for Sphinx build env hash."""

    def __init__(self):
        self.t0 = time.time()

    def __repr__(self):
        return f"<{self.__class__.__name__}>"

    def __call__(self, gallery_conf, fname, when):
        import matplotlib.pyplot as plt

        try:
            from pyvista import Plotter  # noqa
        except ImportError:
            Plotter = None  # noqa
        try:
            from pyvistaqt import BackgroundPlotter  # noqa
        except ImportError:
            BackgroundPlotter = None  # noqa
        try:
            from vtkmodules.vtkCommonDataModel import vtkPolyData  # noqa
        except ImportError:
            vtkPolyData = None  # noqa
        try:
            from mne_qt_browser._pg_figure import MNEQtBrowser
        except ImportError:
            MNEQtBrowser = None
        from mne.viz.backends.renderer import backend

        _Renderer = backend._Renderer if backend is not None else None
        # in case users have interactive mode turned on in matplotlibrc,
        # turn it off here (otherwise the build can be very slow)
        plt.ioff()
        plt.rcParams["animation.embed_limit"] = 40.0
        plt.rcParams["figure.raise_window"] = False
        # https://github.com/sphinx-gallery/sphinx-gallery/pull/1243#issue-2043332860
        plt.rcParams["animation.html"] = "html5"
        # neo holds on to an exception, which in turn holds a stack frame,
        # which will keep alive the global vars during SG execution
        try:
            import neo

            neo.io.stimfitio.STFIO_ERR = None
        except Exception:
            pass
        gc.collect()

        # Agg does not call close_event so let's clean up on our own :(
        # https://github.com/matplotlib/matplotlib/issues/18609
        mne.viz.ui_events._cleanup_agg()
        assert len(mne.viz.ui_events._event_channels) == 0, list(
            mne.viz.ui_events._event_channels
        )

        when = f"mne/conf.py:Resetter.__call__:{when}:{fname}"
        # Support stuff like
        # MNE_SKIP_INSTANCE_ASSERTIONS="Brain,Plotter,BackgroundPlotter,vtkPolyData,_Renderer" make html-memory  # noqa: E501
        # to just test MNEQtBrowser
        skips = os.getenv("MNE_SKIP_INSTANCE_ASSERTIONS", "").lower()
        prefix = ""
        if skips not in ("true", "1", "all"):
            prefix = "Clean "
            skips = skips.split(",")
            if "brain" not in skips:
                _assert_no_instances(mne.viz.Brain, when)  # calls gc.collect()
            if Plotter is not None and "plotter" not in skips:
                _assert_no_instances(Plotter, when)
            if BackgroundPlotter is not None and "backgroundplotter" not in skips:
                _assert_no_instances(BackgroundPlotter, when)
            if vtkPolyData is not None and "vtkpolydata" not in skips:
                _assert_no_instances(vtkPolyData, when)
            if "_renderer" not in skips:
                _assert_no_instances(_Renderer, when)
            if MNEQtBrowser is not None and "mneqtbrowser" not in skips:
                # Ensure any manual fig.close() events get properly handled
                from mne_qt_browser._pg_figure import QApplication

                inst = QApplication.instance()
                if inst is not None:
                    for _ in range(2):
                        inst.processEvents()
                _assert_no_instances(MNEQtBrowser, when)
        # This will overwrite some Sphinx printing but it's useful
        # for memory timestamps
        if os.getenv("SG_STAMP_STARTS", "").lower() == "true":
            import psutil

            process = psutil.Process(os.getpid())
            mem = sizeof_fmt(process.memory_info().rss)
            print(f"{prefix}{time.time() - self.t0:6.1f} s : {mem}".ljust(22))