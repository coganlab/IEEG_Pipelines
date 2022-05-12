from bids import BIDSLayout
from scipy.io import wavfile
import os.path as op


def get_audio(my_layout: BIDSLayout, sub_id, run) -> list:
    events = my_layout.get(subject=sub_id, run=run, suffix="events",
                           extension=".tsv")[0].get_df()
    where_is = events.where(events["trial_type"] == "Audio")
    index = where_is.dropna().index
    return events["stim_file"][index].tolist()


def concat_audio(my_layout: BIDSLayout, waves: list):
    import numpy as np
    data = np.ndarray((0), dtype=np.float32)
    for file in waves:
        full_file = op.join(my_layout.root, "stimuli", file)
        if op.isfile(full_file):
            data = np.concatenate((data, wavfile.read(full_file)[1]),
                                  axis=None)
        else:
            raise IndexError(full_file + " isn't a file")
    return data


if __name__ == '__main__':
    layout = BIDSLayout(r"C:\Users\Jakda\Box\CoganLab\BIDS-1.2-"
                        r"Phoneme_sequencing\BIDS")
    audio = dict()
    for sub_id in layout.get_subjects():
        audio[sub_id] = dict()
        for run in layout.get_runs():
            audiolist = get_audio(layout, sub_id, run)
            audio[sub_id][run] = concat_audio(layout, audiolist)
