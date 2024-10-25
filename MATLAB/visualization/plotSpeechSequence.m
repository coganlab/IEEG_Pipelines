function  [accResults] = plotSpeechSequence(hemisphere,roi)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
colvals = lines(8);
decodeTimeStruct1D = [];
decodeTimeStruct1Dshuffle = [];
decodeTimeStruct1DSequenceshuffle = [];
load(['pooledSubject_Phoneme_Sequencing_Production_car_' hemisphere '_' roi '_z_score_prodelecs_data_v3_Start_syllable_decoded_1D.mat'])
[ax,accResults{1}] = visTimeGenAcc1DCluster(decodeTimeStruct1D,decodeTimeStruct1Dshuffle,pVal2Cutoff=0.05,...
        axisLabel = 'Response',clowLimit = 0,timePad = 0.2,boxPlotPlace=0.6,showPeaks =2,...
        maxVal = -0.3, chanceVal = 0.5,clabel = 'Accuracy/Chance',colval=colvals(4,:),showAccperChance = 1,searchRange=[-1 1]);
hold on;
load(['pooledSubject_Phoneme_Sequencing_Production_car_' hemisphere '_' roi '_z_score_prodelecs_data_v3_Start_phoneme_decoded_1D.mat'])
[~,accResults{2}] = visTimeGenAcc1DCluster(decodeTimeStruct1D(1,:),decodeTimeStruct1DSequenceshuffle(1,:),pVal2Cutoff=0.05,...
    axisLabel = 'Response',clowLimit = 0,timePad = 0.2,boxPlotPlace=0.8,showPeaks = 2,...
    maxVal = -0.2, chanceVal = 0.1111,clabel = 'Accuracy/Chance',colval=colvals(5,:),tileaxis = ax,showAccperChance = 1,searchRange=[-1 1])
% hold on;
% load(['pooledSubject_Phoneme_Sequencing_Production_car_' hemisphere '_' roi '_z_score_prodelecs_data_v2_Start_vcv_decoded_1D.mat'])
% visTimeGenAcc1DCluster(decodeTimeStruct1D(1,:),decodeTimeStruct1Dshuffle(1,:),pVal2Cutoff=0.05,...
%     axisLabel = 'Response',clowLimit = 0,timePad = 0.2,boxPlotPlace=1,showPeaks = 1,...
%     maxVal = -0.1, chanceVal = 0.25,clabel = 'Accuracy/Chance',colval=colvals(6,:),tileaxis = ax,showAccperChance = 1,searchRange=[-1 1])

end