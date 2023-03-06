function [psdAll,f,fftparams] = getPwelch(ieeg,fs,time,tw)


% ieeg - data (channels X timepoints)
% fs - sampling frequency (Hz)
% time - time in seconds (same length as in ieeg)
% tw - time window in seconds to select for psd ([20 180])

timeSelect = time>=tw(1) & time<=tw(2); 
fftparams.winSize = sum(timeSelect)/20; % seconds
fftparams.noverlap = fftparams.winSize/2; % 1.5 seconds
fftparams.nfft = max(256,2^nextpow2(fftparams.winSize)); % 8192 points
[psdAll,f] = pwelch(ieeg(:,timeSelect)',fftparams.winSize,fftparams.noverlap,fftparams.nfft,fs);
% noisePower60 = mean(psdAll(f>=59&f<=61,:).^2,1);
% figure; plot(f,log10(psdAll(:,51))); 

end