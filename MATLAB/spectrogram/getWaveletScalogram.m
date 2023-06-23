function waveSpec = getWaveletScalogram(ieeg, fs, params)
% getWaveletScalogram - Extracts wavelet scalogram using basewave5.
%
% Inputs:
%    ieeg - ECoG data (channels x trials x time)
%    fs - Sampling frequency in Hz
%    params - Optional parameters (struct):
%        - params.fLow: Low frequency range in Hz (default: 2)
%        - params.fHigh: High frequency range in Hz (default: 500)
%        - params.k0: The mother wavelet parameter (wavenumber) (default: 6)
%        - params.waitc: Flag indicating whether to display a waitbar (default: 0)
%
% Output:
%    waveSpec - Structure containing the wavelet scalogram and parameters:
%        - waveSpec.spec: Wavelet scalogram for each channel (cell array of size [1 x channels])
%        - waveSpec.fscale: Frequency scale of the wavelet scalogram (1./period)
%        - waveSpec.params: Parameters used for the wavelet scalogram
%
% Example:
%    ieeg = rand(10, 100, 1000); % Example ECoG data
%    fs = 1000; % Sampling frequency
%    params.fLow = 2; % Low frequency range
%    params.fHigh = 500; % High frequency range
%    params.k0 = 6; % Mother wavelet parameter
%    params.waitc = 0; % Do not display waitbar
%    waveSpec = getWaveletScalogram(ieeg, fs, params); % Extract wavelet scalogram
%


arguments
    ieeg double % ECoG data (channels x trials x time)
    fs double % Sampling frequency in Hz
    params.fLow double = 2 % Low frequency range in Hz (default: 2)
    params.fHigh double = 500 % High frequency range in Hz (default: 500)
    params.k0 double = 6 % Mother wavelet parameter (default: 6)
    params.waitc logical = 0 % Flag to display a waitbar (default: 0)
end

fLow = params.fLow;
fHigh = params.fHigh;
k0 = params.k0;
waitc = params.waitc;

for iChan = 1:size(ieeg, 1)
    [wave, period] = basewave5(squeeze(ieeg(iChan, :, :)), fs, fLow, fHigh, k0, waitc);
    waveSpec.spec{iChan} = permute(abs(wave), [1 3 2]); % Changing format to trials x time x frequency
end

waveSpec.fscale = 1./period;
waveSpec.params = params;

end
