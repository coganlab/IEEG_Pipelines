function [spec, F] = extract_spectrograms_channel(ieeg, AnaParams, ~)
% extract_spectrograms_channel - Extracts the Multitaper spectrograms of the input signal.
%
% Inputs:
%    ieeg - Input signal (trials x time)
%    AnaParams - Analysis parameters
%
% Outputs:
%    spec - Output spectrograms (trials x time x frequency)
%    F - Frequency vector
%
% Example:
%    ieeg = randn(10, 1000); % Example input signal
%    AnaParams.Fs = 1000; % Sample rate
%    AnaParams.Tapers = [1 1]; % Taper parameters
%    AnaParams.fk = [0 100]; % Frequency range
%    AnaParams.dn = 0.5; % Taper size in seconds
%    [spec, F] = extract_spectrograms_channel(ieeg, AnaParams); % Extract spectrograms
%
1

wsize = AnaParams.Tapers(1); % Window size in seconds
fs = AnaParams.Fs; % Sampling frequency
padzero = wsize/2; % Zero padding
fres = AnaParams.Tapers(2); % Frequency resolution in Hertz
dn = AnaParams.dn; % Taper size in seconds
frange = AnaParams.fk; % Frequency range

if fs < 2048
    pad = 2;
else
    pad = 1;
end

% Perform multitaper spectrogram computation
[spec, F] = tfspec(padarray(squeeze(ieeg)', round(padzero*fs), 0, 'both')', [wsize fres], fs, dn, frange, pad, [], [], [], []);
%[spec, F] = tfspec(squeeze(ieeg), [wsize fres], fs, dn, frange, pad, [], [], [], []);

% Remove padded regions
%spec = spec(:, padout+1:end-padout, :);

end
