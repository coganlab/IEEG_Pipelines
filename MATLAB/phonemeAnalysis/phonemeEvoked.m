function [phonEvok,phonAudio,audioAll,meanPhonEvok,pLabelId,pLabel] =  phonemeEvoked(ieeg,micSamp,timeSamp,phBreak,phTime,atw,tw,ptw,fs,fsAudio)
    phUnique = unique(phBreak);
    %aTimeSamp = timeSamp(timeSamp>=0);
    %ieegAll = ieeg(:,:,timeSamp>=0);
    maxAudLength = max(cellfun('length',micSamp));
    audioAll = zeros(size(ieeg,2),2.*maxAudLength);
    aTimeSamp = [0:2*maxAudLength-1];
    for iTrial =1:size(ieeg,2)          
        micTrial = micSamp{iTrial};
        if(length(micTrial)<(2*maxAudLength))
           micTrial = padarray(micTrial',[0 2*(maxAudLength)-length(micTrial)],0,'post');
        end
        audioAll(iTrial,:) = micTrial;
            %length(audioAll(iTrial,:))
            for phInd = 1:5                
                astartsamp = round((phTime(iTrial,phInd)+(atw(1))*fs)*fsAudio/fs);
                astopsamp = round((phTime(iTrial,phInd)+(atw(2))*fs)*fsAudio/fs);
                adataRange = (astopsamp-astartsamp)+1;
                if(iTrial~=1 && phInd~=1)
                    if(adataRange>size(audioPh,3))
                        adataRange = adataRange - 1;
                    end
                    if(adataRange<size(audioPh,3))
                        adataRange = adataRange + 1;
                    end
                end
                find(aTimeSamp>=astartsamp,1);
                find(aTimeSamp>=astartsamp,1)+ adataRange;
               % audioPh(iTrial,phInd,:) = audioAll(iTrial,find(aTimeSamp>=astartsamp,1):find(aTimeSamp>=astartsamp,1)+adataRange);
               audioPh(iTrial,phInd,:) = audioAll(iTrial,aTimeSamp>=astartsamp & aTimeSamp<=astopsamp);
            end
    end
    pLabelMark = 1;
         for phu = 1:length(phUnique)
            [phidR,phidC] = find(strcmp(phBreak,phUnique(phu)));
            for iR = 1:length(phidR)
                phonAudio(pLabelMark,:) = squeeze(audioPh(phidR(iR),phidC(iR),:));                
                pLabelMark = pLabelMark +1;
            end
         end
    for iChan = 1:size(ieeg,1)
        gpowerenv = squeeze(ieeg(iChan,:,:));
        %powerPh = zeros(size(ieeg,2),5,round((tw(2)-tw(1)).*fs));
        powerPh = []; meanPowerPh = [];
        for iTrial =1:size(ieeg,2)        
            for phInd = 1:5
                startsamp = phTime(iTrial,phInd)+round(tw(1)*fs);
                stopsamp = phTime(iTrial,phInd)+round(tw(2)*fs);
                dataRange = stopsamp-startsamp;
                
                pstartsamp = phTime(iTrial,phInd)+round(ptw(1)*fs);
                pstopsamp = phTime(iTrial,phInd)+round(ptw(2)*fs);
                pdataRange = pstopsamp-pstartsamp;
                
                if(iTrial~=1 && phInd~=1)
                    if(dataRange>size(powerPh,3))
                        dataRange = dataRange - 1;
                    end
                    if(dataRange<size(powerPh,3))
                        dataRange = dataRange + 1;
                    end
%                     if(pdataRange>size(powerPh,3))
%                         pdataRange = pdataRange - 1;
%                     end
%                     if(pdataRange>size(powerPh,3))
%                         pdataRange = pdataRange + 1;
%                     end
                end
%                 dataRange
%                 size(powerPh,3)
                timeStartSamp = find(timeSamp>=startsamp,1);
                powerPh(iTrial,phInd,:) = gpowerenv(iTrial,timeStartSamp:timeStartSamp+dataRange-1);
               %powerPh(iTrial,phInd,:) = gpowerenv(iTrial,timeSamp>=startsamp & timeSamp<=stopsamp);
                meanPowerPh(iTrial,phInd) = mean(gpowerenv(iTrial,find(timeSamp>=pstartsamp,1):find(timeSamp>=pstartsamp,1)+pdataRange));
            end
        end
        pLabelMark = 1;
         for phu = 1:length(phUnique)
            [phidR,phidC] = find(strcmp(phBreak,phUnique(phu)));
            for iR = 1:length(phidR)
                phonEvok(iChan,:,pLabelMark) = squeeze(powerPh(phidR(iR),phidC(iR),:));
                meanPhonEvok(iChan,pLabelMark) = meanPowerPh(phidR(iR),phidC(iR));
                pLabelId(pLabelMark) = phu;
                pLabel{pLabelMark} = phUnique{phu};
                pLabelMark = pLabelMark +1;
            end
         end
    end
end