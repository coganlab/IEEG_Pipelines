"""
"This" is my example-script
===========================

This example doesn't do much, it just makes a simple plot
"""
import mne


def fix_annotations(inst):
    """fix SentenceRep events"""
    is_sent = False
    is_bad = False
    annot = None
    no_response = []
    for i, event in enumerate(inst.annotations):
        if event['description'].strip() in ['Audio']:
            if event['duration'] > 1:
                is_sent = True
            else:
                is_sent = False
        if event['description'].strip() not in ['Listen', ':=:']:
            if is_bad:
                trial_type = "BAD "
            elif is_sent:
                trial_type = "Sentence/"
            else:
                trial_type = "Word/"
        else:
            trial_type = "Start/"
            is_bad = False
            if event['description'].strip() in [':=:']:
                cond = "/JL"
            elif 'Mime' in inst.annotations[i + 2]['description']:
                cond = "/LM"
            elif event['description'].strip() in ['Listen'] and \
                    'Speak' in inst.annotations[i + 2]['description']:
                cond = "/LS"
                if 'Response' not in inst.annotations[i + 3]['description']:
                    is_bad = True
                    no_response.append(i)
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
    return no_response
