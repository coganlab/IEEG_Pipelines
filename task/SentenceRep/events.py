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

        # check if sentence or word trial
        if event['description'].strip() in ['Audio']:
            if event['duration'] > 1:
                is_sent = True
            else:
                is_sent = False

        # check if trials co-occur and mark bad
        if i != 0:
            prev = inst.annotations[i-1]
            if prev['onset'] + prev['duration'] > event['onset']:
                annot[i - 1]['description'] = 'bad' + annot[i - 1][
                    'description']
                event['description'] = 'bad' + event['description']
                mne.utils.logger.warn(f"Condition {i-1} and {i} co-occur")

        # check for trial type or bad
        if event['description'].strip() not in ['Listen', ':=:']:
            if is_bad or 'bad' in event['description'].lower():
                trial_type = "bad "
            elif is_sent:
                trial_type = "Sentence/"
            else:
                trial_type = "Word/"
        else:
            # determine trial type
            trial_type = "Start/"
            is_bad = False
            if event['description'].strip() in [':=:']:
                cond = "/JL"
            elif 'Mime' in inst.annotations[i + 2]['description']:
                cond = "/LM"
            elif event['description'].strip() in ['Listen']:
                cond = "/LS"
                if 'Speak' not in inst.annotations[i + 2]['description']:
                    if 'Response' in inst.annotations[i + 2]['description']:
                        mne.utils.logger.warn(f"Early response condition {i}")
                    else:
                        mne.utils.logger.error(
                            f"Speak cue not found for condition #{i} "
                            f"{event['description']}")
                    is_bad = True
                if len(inst.annotations) < i+4:
                    is_bad = True
                    no_response.append(i)
                elif 'Response' not in inst.annotations[i + 3]['description']:
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
