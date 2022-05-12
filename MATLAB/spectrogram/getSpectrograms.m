function [spec,pPerc]= getSpectrograms(ieeg,goodtrials,tw,etw,efw,prtw,pertw,intF,fs,ispermTest)
% Input
% ieeg - channels x trials x time
% goodtrials - cell{channels};  high SNR trials for each channel
% tw - [startTime endTime] time window in seconds
% etw - [startTime endTime] time window for spectrograms in seconds
% efw = [startFrequency stopFrequency] frequency window for spectrograms Hz
% prtw - [startTime endTime] preOnset time window to get significant
% channels
% pstw - [startTime endTime] postOnset time window to get significant
% channels
% fs - sampling frequency
% ispermTest - 0/1; permutation test to get significant channels
% Output
% spec - cell{1 x channels}; spectrograms of each trial for a channel
% p - 1 x channels; p-value from permutation test to check channel
% significance

% if(isempty(goodtrials))
%     goodtrials = 1:size(ieeg,2);
% end
AnaParams.dn=0.05;
AnaParams.Tapers = [.5,10];
AnaParams.fk = [efw(1) efw(2)];
AnaParams.Fs = fs;
%time =linspace(tw(1),tw(2),size(ieeg,3));
%eTime = time>=etw(1)&time<=etw(2);
channelOfInterest =1:size(ieeg,1);
numPerm = 10000;
        for iChan = 1 : length(channelOfInterest)
            iChan
            %idx = i+256*(t-1)
            tic    
            %idx = find(selectedChannels == channelOfInterest(i)); % Index of the selected channels 
            if(isempty(goodtrials))
                trials_g = 1:size(ieeg,2);
            elseif(iscell(goodtrials))
            trials_g = goodtrials{iChan}; % Good trials for the channel
            else
                trials_g = goodtrials;
            end
            [spec{iChan},F] = extract_spectrograms_channel(squeeze(ieeg(iChan,trials_g,:)),AnaParams);
            gammaFreq = F>=intF(1) & F<=intF(2);
            tspec = linspace(tw(1),tw(2),size(spec{iChan},2));
            prtspec = tspec>=prtw(1) & tspec<=prtw(2);
            perctspec = tspec>=pertw(1) & tspec<=pertw(2);
            %protspec = tspec>=protw(1) & tspec<=protw(2);
            if(ispermTest==1)
                meanBase =[]; meanOnsetPercept = []; meanOnsetProd = [];
                for t = 1:length(trials_g)
                    meanBase(t) = mean2(squeeze(spec{iChan}(t,prtspec,gammaFreq)));
                    meanOnsetPercept(t) = mean2(squeeze(spec{iChan}(t,perctspec,gammaFreq)));
                    %meanOnsetProd(t) = mean2(squeeze(spec{iChan}(t,protspec,gammaFreq)));
                end
                pPerc(iChan) = permtest_sk(meanOnsetPercept,meanBase,numPerm);
               % pProd(iChan) = permtest_sk(meanOnsetProd,meanBase,numPerm);
            
            else
                pPerc(iChan) = 0;
              %  pProd(iChan) = 0;
            end
            etspec = tspec>=etw(1) & tspec<=etw(2);
            spec{iChan} = spec{iChan}(:,etspec,:);
        end
end