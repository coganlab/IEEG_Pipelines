function strfChanMap(strfModelFit,stimInfo,chanMap,selectedChannels,isIndView)
F = stimInfo.f;
if(isIndView)
for i = isIndView
    tspec = strfModelFit{1,i}.delays;
    strf2visualize =  squeeze(strfModelFit{1,i}.w1);
    figure;
        %subplot(size(chanMap,1),size(chanMap,2),find(ismember(chanMap',selectedChannels(i))));
        %        imagesc(tspec,F,sq(specMean)');
        if(max(max(abs(strf2visualize))) ==0)
        tvimage((strf2visualize'),'XRange',[tspec(1),tspec(end)],'YRange',[F(1),F(end)]);
        else
            tvimage((strf2visualize'),'XRange',[tspec(1),tspec(end)],'YRange',[F(1),F(end)],'CLim', [-max(max(abs(strf2visualize))) max(max(abs(strf2visualize)))]);
     
        end
        xlabel('Time (ms)');
        ylabel ('Frequency (Hz)');
        colormap(jet(4096));
end
else
     figure;
    for i = 1 : length(strfModelFit)
        tspec = strfModelFit{1,i}.delays;
        strf2visualize =  squeeze(strfModelFit{1,i}.w1);
        subplot(size(chanMap,1),size(chanMap,2),find(ismember(chanMap',selectedChannels(i))));
        %        imagesc(tspec,F,sq(specMean)');
        if(max(max(abs(strf2visualize))) ==0)
        tvimage((strf2visualize'),'XRange',[tspec(1),tspec(end)],'YRange',[F(1),F(end)]);
        else
        tvimage((strf2visualize'),'XRange',[tspec(1),tspec(end)],'YRange',[F(1),F(end)],'CLim', [-max(max(abs(strf2visualize))) max(max(abs(strf2visualize)))]);
        end
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