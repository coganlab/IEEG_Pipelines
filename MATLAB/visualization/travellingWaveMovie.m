function [waveChannel,timeSelect] = travellingWaveMovie(sig2Movie,chanMap,selectedChannels,timeAll,options)

arguments
    sig2Movie double % sig2Movie: channels x timepoints
    chanMap double % chanMap: 2D channel map
    selectedChannels double % selectedChannels: 1 x channels 
    timeAll double % timeAll: 1 x timepoints (in seconds)
    options.etw double = [timeAll(1) timeAll(end)] % etw: epoch time window in seconds (e.g. [-1 1] to print movie
            % between -1 to 1 seconds
    options.clim double = [0 20]  % clim: Colorbar range in uV or z-score value (e.g. [0 20])
    options.frameRate double =  120 % frameRate: Frame rate of the movie (e.g. 120)
    options.movTitle char = 'patient_space_time_activation' % movTitle: Filename to be saved (e.g. 'S23_highGamma')
    options.colbarTitle string = '\muV' % colbarTitle: Color axis label (e.g. 'z-score')
end

        etw = options.etw; 
        clim = options.clim; 
        frameRate = options.frameRate ;
        movTitle = options.movTitle;
        colbarTitle = options.colbarTitle; 
        
%         selectedChannels = sort(chanMap(~isnan(chanMap)))';
        timeSelectInd = timeAll>=etw(1)&timeAll<=etw(2);
        timeSelect = timeAll(timeSelectInd);
        figure;
        plot(timeSelect,sig2Movie(:,timeSelectInd),'color',[0 0 0] +0.75);
        hold on;
        plot(timeSelect,mean(sig2Movie(:,timeSelectInd),1),'color',[0 0 0]);
        ylim(clim);
        xlabel('Time (s)');
        ylabel(colbarTitle);
        title(movTitle);
        saveas(gcf,[movTitle '_timeSeries.png']);
        waveChannel = nan(size(chanMap,1),size(chanMap,2),length(timeAll(timeSelectInd)));
        for c = 1 : length(selectedChannels)
            [cIndR, cIndC] = find(ismember(chanMap,selectedChannels(c)));
            for ind=1:length(cIndR)
                waveChannel(cIndR(ind),cIndC(ind),:)=sig2Movie(c,timeSelectInd);
            end
        end
        figure;
        for iTime=1:size(waveChannel,3)
          %  surfc(X,Y,sq(spec_chansBHG(:,:,iT)),'FaceAlpha',0.5);
            b = imagesc(sq(waveChannel(:,:,iTime))); 
            cb = colorbar;
            ylabel(cb,colbarTitle)
            %truesize(gcf,[1000 500]);
            set(b,'AlphaData',~isnan(sq(waveChannel(:,:,iTime))));
            caxis([clim(1) clim(2)])          
            set(gca,'xtick',[])
            set(gca,'xticklabel',[])
            set(gca,'ytick',[])
            set(gca,'yticklabel',[])        
            axis equal
            axis tight
            set(gca,'FontSize',20);
            colormap(jet(4096))
            
         title([num2str(round(timeSelect(iTime),2)) ' s'])
           M(iTime)=getframe(gcf);
        end
        cmap=colormap('jet');
        close
        vname = strcat(movTitle,'.avi');
        vidObj=VideoWriter(vname, 'Motion JPEG AVI');
        vidObj.Quality = 100;    
        vidObj.FrameRate = frameRate;
        open(vidObj);        
        writeVideo(vidObj,M);
         close(vidObj);
end