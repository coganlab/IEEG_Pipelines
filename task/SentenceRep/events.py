import mne


def fix_annotations(inst):
    # fix SentenceRep events
    is_sent = False
    annot = None
    for i, event in enumerate(inst.annotations):
        if event['description'] in ['Audio']:
            if event['duration'] > 1:
                is_sent = True
            else:
                is_sent = False
        if event['description'] not in ['Listen', ':=:']:
            if is_sent:
                trial_type = "Sentence/"
            else:
                trial_type = "Word/"
        else:
            trial_type = "Start/"
            if event['description'] in [':=:']:
                cond = "/JL"
            elif 'Mime' in inst.annotations[i + 2]['description']:
                cond = "/LM"
            elif event['description'] in ['Listen'] and \
                    'Response' in inst.annotations[i + 3]['description']:
                cond = "/LS"
            else:
                raise ValueError("Condition {} could not be determined {}"
                                 "".format(i, event['description']))

        event['description'] = trial_type + event['description'] + cond
        if annot is None:
            annot = mne.Annotations(**event)
        else:
            event.pop('orig_time')
            annot.append(**event)
    inst.set_annotations(annot)

# events, event_id = mne.events_from_annotations(good)
# events = mne.merge_events(events, [1, 2], 14, replace_events=True)
# event_id['Start'] = 14
# event_id = dict([(value, key) for key, value in event_id.items()])
# annot = mne.annotations_from_events(events, filt.info['sfreq'], event_id)
# filt.set_annotations(annot)
