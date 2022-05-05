function [meanIeegRaw,meanIeegRawNorm, meanIeegGammaPowerNorm, timeInterest] = travelling_wave_analysis(ieeg,goodTrials,selectedChannels)
    %load('goodTrials.mat');
    fs = 2000;
    %dt = 1/fs;    
    %cCount = 1024;
    cdiv = length(selectedChannels);
    rawF = [1 59];
    gammaF = [70 160];
    pgfilt = gausswin(0.05*fs);
    time = linspace(-1,5,6*fs);
    preTime = time>=-0.5 & time<=0;
    timeInterestId = time>=-0.5 & time<=4.5;
    timeInterest = time(timeInterestId);
%     for t = 1:4
%         ieegstruct = load(strcat('ieegCM-1split5',num2str(t),'.mat'));
%         fname = fieldnames(ieegstruct);
%         ieeg2analyze = getfield(ieegstruct,fname{1});
        for i = 1:cdiv
            %idx = i+256*(t-1);
            trials_g = goodTrials{i};
            sig2Analyze = squeeze(ieeg(i,trials_g,:));
            ieegRaw = eegfilt(sig2Analyze,fs,rawF(1),rawF(2),0,200);
            ieegGamma = eegfilt(sig2Analyze,fs,gammaF(1),gammaF(2),0,200);   
            ieegGammaPowerTrial = (abs(hilbert(ieegGamma)).^2); 
            parfor tr = 1:length(trials_g)
                ieegGammaPowerTrial(tr,:) = log(filtfilt(pgfilt,1,ieegGammaPowerTrial(tr,:)));%    
                %ieegRawSmooth(tr,:) = filtfilt(pgfilt,1,ieegRaw(tr,:));
            end
            ieegGammaPowerTrialNorm = (ieegGammaPowerTrial - mean2(ieegGammaPowerTrial(:,preTime)))./std2(ieegGammaPowerTrial(:,preTime));
            ieegRawTrialNorm = (ieegRaw - mean2(ieegRaw(:,preTime)))./std2(ieegRaw(:,preTime));
           % ieegRawTrialNorm = (ieegRaw - mean(ieegRaw(:,preTime),2));
            meanIeegRawNorm(i,:) = mean(ieegRawTrialNorm(:,timeInterestId),1);
            meanIeegRaw(i,:) = mean(ieegRaw(:,timeInterestId),1);
            meanIeegGammaPowerNorm(i,:) = mean(ieegGammaPowerTrialNorm(:,timeInterestId),1);            
        end        
    
end