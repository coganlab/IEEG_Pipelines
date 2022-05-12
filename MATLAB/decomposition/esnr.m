function evokeSnr = esnr(chanSignal,chanNoise)
    sigmaNoise = cov1para(chanNoise);
    mDistNoise = mahalUpdate(chanNoise,chanNoise,sigmaNoise);
    varNoise = exp(mean(log(mDistNoise.^2)));
    mDistSignal = mahalUpdate(chanSignal,chanNoise,sigmaNoise);
    evokeSnr = 10.*log10(mean(mDistSignal.^2/varNoise));
end