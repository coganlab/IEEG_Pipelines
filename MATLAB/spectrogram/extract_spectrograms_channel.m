function [spec,F] = extract_spectrograms_channel(ieeg, AnaParams,~)
% Extracts the Multitaper spectrograms
% INPUT
%       ieeg - Input signal (trials x time)
%       AnalParams - Analysis parameters; 
% OUTPUT
%       spec - Output Spectrograms (trials x time x frequency)                    
wsize = AnaParams.Tapers(1); % Window size in seconds
fs = AnaParams.Fs; % Sampling frequency
padzero = wsize/2; % Zero padding
fres = AnaParams.Tapers(2); % Frequency resolution in Hertz
dn = AnaParams.dn; % Taper size in seconds
frange = AnaParams.fk; % Frequency range
if fs<2048
    pad=2;
else
    pad=1;
end
[spec,F]=tfspec(padarray(squeeze(ieeg)',round(padzero*fs),0,'both')',[wsize fres],fs,dn,frange,pad,[],[],[],[]);
%[spec,F]=tfspec(squeeze(ieeg),[wsize fres],fs,dn,frange,pad,[],[],[],[]);
%spec = spec(:,padout+1:end-padout,:);