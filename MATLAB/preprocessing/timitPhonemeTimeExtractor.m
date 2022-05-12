function [phonMet,gammaAll,pLabel] =  timitPhonemeTimeExtractor(audioPathUsed,micAudio,ieegGamma,etw,pSpan)
    timitPath = 'E:\timit\TIMIT\';
    
    timeGamma =linspace(etw(1),etw(2),size(ieegGamma,3));
    phonMet = []; gammaAll = []; pLabel = [];audioAll = [];
    for iAudio = 1:length(audioPathUsed)
       phonMetTemp = [];
        sentPath = [timitPath audioPathUsed{iAudio}];
        [audFile,fsAudio] = audioread([timitPath audioPathUsed{iAudio}]);
        phonPath = [sentPath(1:end-3) 'PHN'];        
        [pb,pe,pname] = textread([phonPath],'%n %n %s');
        phonMetTemp.pb = pb./fsAudio;
        phonMetTemp.pe = pe./fsAudio;
        phonMetTemp.pname = pname;
        if(iAudio==1)
            timeAudio = [0:size(micAudio,2)-1]./20000;
            timeAudioRead = [0:length(audFile)-1]./fsAudio;
            figure;
            plot(timeAudio,micAudio(iAudio,:));
            hold on;
            plot(timeAudioRead,audFile);
            scatter(pb'./fsAudio,0.2*ones(size(pb')));
            
            
        end
        %audFile = audFile';
        for iPhon = 1:length(pb)

            phonMetTemp.gamma(:,:,iPhon) = ieegGamma(:,iAudio,timeGamma>=(round(pb(iPhon)./fsAudio+pSpan(1),2)) & timeGamma<=(round(pb(iPhon)./fsAudio+pSpan(2),2)));

        end
        gammaAll = cat(3,gammaAll,phonMetTemp.gamma);
      
        pLabel = [pLabel; phonMetTemp.pname];
    
    end
end