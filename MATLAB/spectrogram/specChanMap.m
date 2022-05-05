function [specMeanAll,specMeanPower,meanFreqChanOut] = specChanMap(spec,chanMap,selectedChannels,sigChannel,tw,entw,etw,efw,gammaF,cval,isIndView,meanFreqChanIn)
tspec = linspace(tw(1),tw(2),size(spec{1},2));
F = linspace(efw(1),efw(2),size(spec{1},3));
meanFreqChanOut = [];
if(isIndView)
        for iChan = isIndView
            spec2Analyze = spec{iChan};
           % spec2Analyze = spec{i};
           meanFreq = [];
           if(isempty(meanFreqChanIn))
                for iFreq = 1:size(spec2Analyze,3)
                    meanFreq(iFreq) = mean2(squeeze((spec2Analyze(:,tspec>=entw(1)&tspec<= entw(2),iFreq))));
                end
                meanFreqChanOut(iChan,:) = meanFreq;
           else
               meanFreq = meanFreqChanIn(iChan,:);
           end
            specMean = squeeze(mean(spec2Analyze./reshape(meanFreq,1,1,[]),1));
                specMeanAll(iChan,:,:) = specMean';
                specMeanPower = mean2(squeeze(specMeanAll(iChan,F>=gammaF(1)&F<=gammaF(2),tspec>=etw(1)&tspec<=etw(2))));

           
            imagesc(tspec,F,20.*log10(sq(specMean)'))
%             if(sum(ismember(sigChannel,iChan)))
%                     axis on;
%                     set(gca,'linewidth',2)
%                     ax = gca;
%                     ax.XColor = 'black';
%                     ax.YColor = 'black';
%             end
            %tvimage(sq(specMean),'XRange',[tspec(1),tspec(end)],'YRange',[F(1),F(end)],'CLim',cval);
        %     xlabel('Time (s)');
        %     ylabel('Frequency (Hz)');
            if(isnumeric(selectedChannels(iChan)))
                title(strcat('Channel: ',num2str(selectedChannels(iChan))));
            else
                 title(selectedChannels(iChan));
            end
            colormap(parula(4096));
            if(~isempty(cval))
            caxis(cval);
            end
           set(gca,'YDir', 'normal');
           %figure;
           %plot(tspec,mean(sq(specMean(:,F>=70&F<=150)),2));
        end
else
    %figure;
    specMeanAll = [];
    for iChan = 1 : length(spec)
         spec2Analyze = spec{iChan};
       % spec2Analyze = spec{i};
       meanFreq = [];
         if(isempty(meanFreqChanIn))
                for iFreq = 1:size(spec2Analyze,3)
                    meanFreq(iFreq) = mean2(squeeze((spec2Analyze(:,tspec>=entw(1)&tspec<= entw(2),iFreq))));
                end
                meanFreqChanOut(iChan,:) = meanFreq;
           else
               meanFreq = meanFreqChanIn(iChan,:);
         end

        specMean = squeeze(mean(spec2Analyze./reshape(meanFreq,1,1,[]),1));
        specMeanAll(iChan,:,:) = specMean';
        specMeanPower(iChan) = mean2(squeeze(specMeanAll(iChan,F>=gammaF(1)&F<=gammaF(2),tspec>=etw(1)&tspec<=etw(2))));
        
        [p]=numSubplots(length(spec));
        if(isempty(chanMap))
            
            subaxis(p(1),p(2),iChan,'sh', 0.03, 'sv', 0.01, 'padding', 0, 'margin', 0)
            imagesc(tspec,F,20.*log10(sq(specMean)'));
            axis off;
            axis('square');
            axis tight;
%             ax = gca;
%             ax.Position = [ax.Position(1) ax.Position(2) 1/p(2) 1/p(1)];
        else
            subaxis(size(chanMap,1),size(chanMap,2),find(ismember(chanMap',selectedChannels(iChan))),'SpacingHoriz', 0.0001/size(chanMap,2), 'SpacingVert', 0.1/size(chanMap,2), 'padding', 0, 'margin', 0);
            imagesc(tspec,F,20.*log10(sq(specMean)'));
            axis off;
            axis square;
            axis tight;
%             ax = gca;
%             ax.Position = [ax.Position(1) ax.Position(2) 0.25/size(chanMap,2) 0.75/size(chanMap,1)];
        end
        
        %tvimage(sq(specMean),'XRange',[tspec(1),tspec(end)],'YRange',[F(1),F(end)],'CLim',cval);
        
        if(sum(ismember(sigChannel,iChan)))
            axis on;
            set(gca,'linewidth',3)
            ax = gca;
            ax.XColor = 'black';
            ax.YColor = 'black';
        end
%         if(isnumeric(selectedChannels(iChan)))
%                 title(strcat('Channel: ',num2str(selectedChannels(iChan))));
%             else
%                  title(selectedChannels(iChan));
%         end
        set(gca,'xtick',[],'ytick',[])
        colormap(parula(4096));
        if(~isempty(cval))
            caxis(cval);
        end
       set(gca,'YDir', 'normal');
    end
end

end
