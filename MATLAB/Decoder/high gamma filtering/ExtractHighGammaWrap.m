function [ieegGammaNorm,ieegGamma,p_masked] = ExtractHighGammaWrap(ieegSplit,fsIn,fsOut,tw,etw,prtw,pstw)
% The function extracts Normalized high-gamma using Nima Mesgarani's
% algorithm
% Input
% ieegSplit (channels x trials x time-points): Input raw data
% fsIn: Input sampling frequency in Hz.
% fsOut: Output sampling frequency in Hz.
% tw: Input time window in seconds
% etw: Output time window in seconds (Range should be within 'tw' to remove
% edge artifacts
% prtw: baseline time-window (seconds)
% pstw: response time-window (seconds
% Output
% ieegGammaNorm (channels x trials x time-points): Normalized high-gamma
% ieegGamma (channels x trials x time-points): high-gamma envelope
% p_masked (channels): significant channels

fGamma = [70 150]; % High gamma range
ieegGamma = [];
% High Gamma extraction
disp('Extracting High Gamma');
for iTrial = 1:size(ieegSplit,2)
      
    [~,ieegTmp] = EcogExtractHighGammaTrial(double(squeeze(ieegSplit(:,iTrial,:))),fsIn,fsOut,fGamma,tw,etw,[],[]);
    ieegGamma(:,iTrial,:) = ieegTmp;   
end
timeGamma = linspace(etw(1),etw(2),size(ieegGamma,3));
ieegGammaResponse = squeeze(mean(ieegGamma(:,:,timeGamma>=pstw(1)&timeGamma<=pstw(2)),3)); % channels x trials
ieegGammaBase = squeeze(mean(ieegGamma(:,:,timeGamma>=prtw(1)&timeGamma<=prtw(2)),3));% channels x trials
pChan = [];
% Identifying significant channels based on high-gamma power
for iChan = 1:size(ieegSplit,1)
    pChan(iChan) = permtest_sk(ieegGammaResponse(iChan,:),ieegGammaBase(iChan,:),10000);
end
[p_fdr, p_masked] = fdr( pChan, 0.05);
disp(['Number of significant channels : ' num2str(sum(p_masked))]);
% Extracting normalizing factors for each channel
normFactorIntTrial = [];
for iChan = 1:size(ieegSplit,1)
    normFactorIntTrial(iChan,:) = [mean2(squeeze(ieegGamma(iChan,:,timeGamma>=prtw(1)&timeGamma<=prtw(2)))) std2(squeeze(ieegGamma(iChan,:,timeGamma>=prtw(1)&timeGamma<=prtw(2))))];
end
% Normalized high gamma extraction
disp('Extracting Normalized High Gamma');
ieegGammaNorm = [];
for iTrial = 1:size(ieegSplit,2)
      
    [~,ieegTmp] = EcogExtractHighGammaTrial(double(squeeze(ieegSplit(:,iTrial,:))),fsIn,fsOut,fGamma,tw,etw,normFactorIntTrial,1);
    ieegGammaNorm(:,iTrial,:) = ieegTmp;   
end
end