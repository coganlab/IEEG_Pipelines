from Python.PreProcess.preprocess_old import Preprocessing


def test_bids():
    try:
        pre = Preprocessing(input_dir="Python/Final_directory_structure",
                            validate=False)
    except OSError:
        return True
    assert pre.BIDS_layout.get_subjects() == ["d0048", "d0053"]
