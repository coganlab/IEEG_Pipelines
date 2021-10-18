from bids import BIDSLayout

def get_audio(my_layout: BIDSLayout, sub_id, run) -> list:
    events = my_layout.get(subject=sub_id, run=run, suffix="events",
                        extension=".tsv")[0].get_df()
    where_is = events.where(events["trial_type"] == "Audio")
    index = where_is.dropna().index
    return events["stim_file"][index].tolist()

#def concat_audio(waves: list, )
if __name__ == '__main__':
    layout = BIDSLayout(r"C:\Users\Jakda\Box\CoganLab\BIDS-1.2-"
                        r"Phoneme_sequencing\BIDS")
    audio = dict()
    for sub_id in layout.get_subjects():
        audio[sub_id] = dict()
        for run in layout.get_runs():
            audio[sub_id][run] = get_audio(layout, sub_id, run)

