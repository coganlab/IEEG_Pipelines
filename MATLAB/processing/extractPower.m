function [ieegPowerSeries,ieegFilt,ieegPower] = extractPower(ieegSplit,fs,Frange,tw,etw)
    for iChan = 1 : size(ieegSplit,1)
        ieegFiltChannel = eegfilt(squeeze(ieegSplit(iChan,:,:)),fs,Frange(1),Frange(2),0,200);
        %ieegFiltChannel = squeeze(ieegSplit(i,:,:));
        ieegFilt(iChan,:,:) = ieegSplit(iChan,:,:);
        time = linspace(tw(1),tw(2),size(ieegSplit,3));
        for iTrial = 1:size(ieegSplit,2)
            %ieegPower(i,tr) = mean(log10(ieegFiltChannel(tr,time>=tw(1)&time<=tw(2)).^2));
            ieegPower(iChan,iTrial) = mean((ieegFiltChannel(iTrial,time>=etw(1)&time<=etw(2))));
            ieegPowerSeries(iChan,iTrial,:) = (abs(hilbert(ieegFiltChannel(iTrial,:))));
        end
    end    
end