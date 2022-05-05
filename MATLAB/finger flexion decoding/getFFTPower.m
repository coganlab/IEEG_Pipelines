function [sigPower,sigFilt] = getFFTPower(ieegSplit,fs,tw,etw,fband)

pfilt=designfilt('bandpassiir',...       % band pass filter for beta extraction
    'FilterOrder',8,'PassbandFrequency1',fband(1),'PassbandFrequency2',fband(2),...
    'PassbandRipple',0.2,'SampleRate',fs);
time = linspace(tw(1),tw(2),size(ieegSplit,3));
eTimeWindow = time>=etw(1)&time<=etw(2);
for iChan = 1:size(ieegSplit,1)
    for iTrial = 1:size(ieegSplit,2)
        sigFiltTemp = filtfilt(pfilt,squeeze(ieegSplit(iChan,iTrial,:)));
        sigPower(iChan,iTrial) = mean(log10(sigFiltTemp(eTimeWindow).^2));
        sigFilt(iChan,iTrial,:) = sigFiltTemp(eTimeWindow);
    end
end
end