function timeSeriesChanMap(sig2view,chanMap,selectedChannels,timeInterest,yval,etw,label)
if(~isempty(label))
    
    sgtitle(label)
else
end
if(length(size(sig2view))==2)
    for iChan = 1 : size(sig2view,1)

        subplot(size(chanMap,1),size(chanMap,2),find(ismember(chanMap',selectedChannels(iChan))));
        %
        plot(timeInterest(timeInterest>=etw(1)&timeInterest<=etw(2)),sig2view(iChan,timeInterest>=etw(1)&timeInterest<=etw(2)),'LineWidth',2);
    %     
        if(~isempty(yval))
        ylim(yval)
        end
    
    end
else
    for iChan = 1 : size(sig2view,2)

        subplot(size(chanMap,1),size(chanMap,2),find(ismember(chanMap',selectedChannels(iChan))));
        %
        for iTrial = 1:size(sig2view,1)
            plot(timeInterest(timeInterest>=etw(1)&timeInterest<=etw(2)),squeeze(sig2view(iTrial,iChan,timeInterest>=etw(1)&timeInterest<=etw(2))),'LineWidth',2);
            hold on;
        end
        %     hold on;
    %     vline(0,'r','')
        if(~isempty(yval))
        ylim(yval)
        end
    %     axis off;
         set(gca,'xtick',[])    
         set(gca,'ytick',[])
    end
end
% if(~isempty(title))
%     sgtitle(title)
% end
end