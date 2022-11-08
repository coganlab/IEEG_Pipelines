function SBP = EcogExtractSpikeBandPowerTrial(data, inFs, fRange, tw, etw, nFilter, decimateFactor)
% Gets spike band power of neural signal 
% 
% Extracts spike band power (SBP) of neural signal as defined in the 
% methods of https://www.nature.com/articles/s41551-020-0591-0 
% (Nason et al. 2020). Data is filtered between 300-1000 Hz, rectified,
% convolved with a 50 ms Gaussian kernel, and downsampled to desired freq.
% Incoming data usually comes in at 20 kHz. Downsampling should be done to
% >= 2000 Hz to prevent anti-aliasing of SBP signal
%
% data (matrix): recorded data, channel x (timepoints = inFS*(tw(2)-tw(1)))
% inFs (int): sampling rate of data
% fRange ([int int]): bandpass filtering range in Hz e.g. [300 1000]
% tw ([double double]): time-window of data e.g. [-2 2]
% etw ([double double]): output time-window e.g. [-1 1]
% nFilter (int): Order of bandpass filter (default = 200)
% decimateFactor (int): factor to decimate signal by (default=10 to
%                       downsample from 20 kHz to 2 Hz
% 
% SBP (matrix): spike band power of input data,  
%               channel x (timepoints = outFs*(etw(2) - etw(1)))

SBP = eegfilt(data,inFS,fRange(1),fRange(2),0,nFilter);

SBP = abs(SBP); % rectify signal

% SBP smoothing
winLenSBP = 0.050; % seconds
gaussWinSBP = gausswin(winLenSBP*inFs);
SBP = conv(SBP, gaussWinSBP, 'same'); % smooth with 50 ms Gaussian window

% downsample filtered data and get SBP peaks
SBP = decimate(SBP, decimateFactor); % 20,000 Hz -> 2,000 Hz
end

