function waveSpec = getWaveletScalogram(ieeg,fs,params)
%GETWAVELETSCALOGRAM extract wavelet scalogram using basewave5
%   
arguments
    ieeg double % ieeg data channels x trials x time
    fs double % sampling frequency in Hz
    params.fLow double  = 2 % low frequency range in Hz
    params.fHigh double = 500 % high frequency range in Hz
    params.k0 double = 6 % the mother wavelet parameter (wavenumber), default is 6.
    params.waitc logical = 0 % a handle to the qiqi waitbar.
end
for iChan = 1:size(ieeg,1)
    [wave,period]=basewave5(squeeze(ieeg(iChan,:,:)),fs,params.fLow,params.fHigh,params.k0,params.waitc);
    waveSpec.spec{iChan} = abs(wave);
end
waveSpec.fscale = 1./period;
waveSpec.params = params;


end

