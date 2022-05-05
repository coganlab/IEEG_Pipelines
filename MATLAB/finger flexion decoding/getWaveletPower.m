function sigPower = getWaveletPower(ieegSplit,fs,tw,etw,fband)
opts = struct('type','fft','fs',fs,'sampling','freq','fmin',5,'fmax',200);
time = linspace(tw(1),tw(2),size(ieegSplit,3));
eTimeWindow = time>=etw(1)&time<=etw(2);
for iChan = 1:size(ieegSplit,1)
    parfor iTrial = 1:size(ieegSplit,2)
        [wavetrans,fscale]=wt(squeeze(ieegSplit(iChan,iTrial,:)),opts);
        wavetransPower=log10(wavetrans.^2);
        sigPower(iChan,iTrial) = mean2(wavetrans(fscale>=fband(1)&fscale<=fband(2),eTimeWindow));
    end
end

end