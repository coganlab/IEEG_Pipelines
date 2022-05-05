function specChanMapNoNorm(spec,chanMap,selectedChannels,tw,fw,etw,efw,cval,isIndView)
tspec = linspace(tw(1),tw(2),size(spec{1},2));
Fspec = linspace(fw(1),fw(2),size(spec{1},1));
etspec = find(tspec>=etw(1)&tspec<=etw(2));
efspec = find(Fspec>=efw(1)&Fspec<=efw(2));
if(isIndView)
for i = isIndView
    %spec2Analyze = spec{i}./mean(mean(spec{i}(:,tspec>=-0.5&tspec<=0,:),2),1);
    specMean = spec{i}(efspec,etspec);
     imagesc(tspec(etspec),Fspec(efspec),specMean);
    caxis(cval);
    colormap(jet(4096));
    set(gca,'YDir', 'normal');
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    title(strcat('Channel: ',num2str(selectedChannels(i))));
    colormap(jet(4096));
%     if(~isempty(cval))
%     caxis(cval);
%     end
 %   set(gca,'YDir', 'normal');
end
    else
    figure;
    for i = 1 : length(spec)
        %spec2Analyze = spec{i}./mean(mean(spec{i}(:,tspec>=-0.5&tspec<=0,:),2),1);
        specMean = spec{i}(efspec,etspec);
        subplot(size(chanMap,1),size(chanMap,2),find(ismember(chanMap',selectedChannels(i))));
        imagesc(tspec(etspec),Fspec(efspec),specMean);
        caxis(cval);
        colormap(jet(4096));
        set(gca,'YDir', 'normal');
        axis off;
        set(gca,'xtick',[],'ytick',[])
        colormap(jet(4096));
%         if(~isempty(cval))
%             caxis(cval);
%         end
     %   set(gca,'YDir', 'normal');
    end
end

end
