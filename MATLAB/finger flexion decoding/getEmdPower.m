function sigPower = getEmdPower(ieegSplit,tw,etw,numIMF,imfNo)
time = linspace(tw(1),tw(2),size(ieegSplit,3));
eTimeWindow = time>=etw(1)&time<=etw(2);
Nstd=0; % Noise standard deviation for EEMD; if Nstd=0, run EMD
NE=100; % Number of ensembles for EEMD;
for iChan = 1:size(ieegSplit,1)
    for iTrial = 1:size(ieegSplit,2)      
        imfECoG = feemd(squeeze(ieegSplit(iChan,iTrial,:)),Nstd,NE,numIMF);
        sigFilt = imfECoG(imfNo,:);
        sigPower(iChan,iTrial) = mean(log10(sigFilt(eTimeWindow).^2));
    end
end
end