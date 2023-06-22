function ieegSplit = splitIeeg(ieeg, trigOnset, tw, fs)
% splitIeeg - Splits the iEEG data into individual trials based on trigger onsets.
%
% Syntax: ieegSplit = splitIeeg(ieeg, trigOnset, tw, fs)
%
% Inputs:
%   ieeg        - iEEG data (Channels x Samples)
%   trigOnset   - Trigger onsets indicating the start of each trial (in seconds)
%   tw          - Time window for each trial (in seconds)
%   fs          - Sampling frequency (in Hz)
%
% Outputs:
%   ieegSplit   - Split iEEG data into individual trials (Channels x Trials x Samples)
%


ieegSplit = []; % Initialize the split iEEG data

for iTrial = 1:length(trigOnset)
    iTrial
    ieegSplit(:, iTrial, :) = ieeg(:, round(tw(1) * fs) + trigOnset(iTrial) : trigOnset(iTrial) + round(fs * tw(2)));
end

ieegSplit = ieegSplit(:, :, 1:end-1); % Remove the last sample from each trial

end
