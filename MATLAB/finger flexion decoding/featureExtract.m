% Place the following folders under Software tool in path 
%    a)"EMD",
%    b)"wavelet-master"
% load day2run2_Isometric_Left.mat in UM-01-wednesday
% load day2run2_Isometric_Left.mat in UM-01-Thursday
% load z_iso_right_updated.mat in UM-03
%z = z_right; % only in UM-03 - brutal code

fsTrial = 1000; % sampling frequency of the sensor
fsECoG = 10000; % sampling frequency of ECoG
fsECoGd = fsECoG; % downsampled sampling frequency
tw = [-0.25 3.5]; % time-window across finger movement onset in seconds
% Place the following folders under Software tool in path 
%    a)"EMD",
%    b)"wavelet-master"
% load day2run2_Isometric_Left.mat in UM-01-wednesday
% load day2run2_Isometric_Left.mat in UM-01-Thursday
% load z_iso_right_updated.mat in UM-03
z = z_right; % only in UM-03 - brutal code
% should work in all z struct files

% UM -01 channels
% c_channel1=[56:65]; % cable 1
% c_channel2=[66:75]; % cable 2
% UM -03 channels
c_channel1=[1:35]; % cable 1
c_channel2=[36:41]; % cable 2

trialInt = find(ismember([z.TaskNumber],[1 2 5])); % extracting finger trials only
zInt = z(trialInt);
trialId = [zInt.TaskNumber]; % finger ids
trialOn = [zInt.MoveOnset]; % flexion onsets
%trialOff = [zInt.MoveOffset]; % flexion offset

ieegSplit = []; % channels x trials x timepoints
tc = 1;
emptyTrials = [];
% Processing each trial information
% 1. Downsampling to 2 kHz
% 2. Filtering 60 Hz and its harmonics
% 3. Creating ieeg trial structure
for iTrial = 1:size(zInt,2)
    iTrial
   
    ieegTemp = zInt(iTrial).ECoG;
    if(isempty(ieegTemp))
        emptyTrials = [emptyTrials iTrial];
    continue;
    end
    ieegTemp = cell2mat(squeeze(struct2cell(ieegTemp)));
    
    ieegTemp = resample(ieegTemp',fsECoGd,fsECoG)'; % downsampling
    ieegTempCar1 = ieegTemp(c_channel1,:)-mean(ieegTemp(c_channel1,:),1);
    ieegTempCar2 = ieegTemp(c_channel2,:)-mean(ieegTemp(c_channel2,:),1);
    ieegCar = [ieegTempCar1; ieegTempCar2];
    ieegFilt = ieegCar; % no 60 Hz filtering
    timeTrial = linspace(0,(size(ieegCar,2)-1)./fsECoGd,size(ieegCar,2));
    timeId = find(timeTrial>=trialOn(iTrial)/fsTrial,1);
    ieegTempArrange = ieegFilt(:,timeId+tw(1)*fsECoGd:timeId+tw(2)*fsECoGd);
    ieegSplit(:,tc,:) = ieegTempArrange;
    tc = tc+1;
end
trialId(emptyTrials) = [];
sigChan = [7,8,12,13]; %UM - 01
sigChan = [1,9]; % UM - 03;
ieegSplitSig = ieegSplit(sigChan,:,:);
sigPowerFFT = getFFTPower(ieegSplitSig,fsECoGd,tw,[0 1],[66 114]);
sigPowerEMD = getEmdPower(ieegSplitSig,tw,[0 1],9,3);
sigPowerWav = getWaveletPower(ieegSplitSig,fsECoGd,tw,[0 1],[66 114]);
sigPowerPsd = getPsd(ieegSplitSig,fsECoGd,tw,[0 1],[2 3]);
%% Naive - Bayes Classification

ypredFFT = nbClassify(sigPowerFFT',double(trialId),1);
accFFT = sum(ypredFFT==trialId)/length(trialId)
ypredEMD = nbClassify(sigPowerEMD',double(trialId),1);
accEMD = sum(ypredEMD==trialId)/length(trialId)
ypredWav = nbClassify(sigPowerWav',double(trialId),1);
accWav = sum(ypredWav==trialId)/length(trialId)
 ypredPca = nbClassifyPSCA(sigPowerPsd(:,1:length(trialId),:),fsECoGd,double(trialId),1,1);
 accPca = sum(ypredPca==trialId)/length(trialId)