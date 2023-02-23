import mne


def fix_annotations(inst):
    """fix SentenceRep events"""
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
