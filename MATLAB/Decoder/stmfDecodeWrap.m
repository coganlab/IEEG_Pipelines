function [accAll,ytestAll,ypredAll,aucAll] = stmfDecodeWrap(ieegSplit,labels,tw,etw,numFolds)
% The function performs supervised spatio-temporal decoding on ephys time-series dataset;
% Step 1: Spatio-temporal template estimation from the training set
% Step 2: Decoding using correlation analysis
%
% Input
% ieegSplit (channels x trials x time): Input dataset
% labels (trials): Input labels
% tw: epoch time-window
% etw: selected time-window for decoding
% % numFolds: Number of folds for cross-validation
% Output
% accAll - overall accuracy (0 - 1)
% ytestAll - tested labels
% ypredall - predicted labels
% aucAll - ROC - AUC
% Reference : Ramsey NF, Salari E, Aarnoutse EJ, Vansteensel MJ, Bleichner MG, Freudenburg ZV. 
% Decoding spoken phonemes from sensorimotor cortex with high-density ECoG grids. 
% Neuroimage. 2018 Oct 15;180(Pt A):301-311. doi: 10.1016/j.neuroimage.2017.10.011. 
% Epub 2017 Oct 7. PMID: 28993231; PMCID: PMC6433278.


timeSplit = linspace(tw(1),tw(2),size(ieegSplit,3));
timeSelect = timeSplit>=etw(1)&timeSplit<=etw(2);
ieegModel = ieegSplit(:,:,timeSelect);

accAll = 0;

ytestAll = [];
ypredAll = [];

if(numFolds>0)
    cvp = cvpartition(labels,'KFold',numFolds,'Stratify',true);
else
    cvp = cvpartition(labels,'LeaveOut');
end


for nCv = 1:cvp.NumTestSets
        % Training - Testing Split
        train = cvp.training(nCv);
        test = cvp.test(nCv);
        ieegTrain = ieegModel(:,train,:);
        ieegTest = ieegModel(:,test,:);
        matTrain = size(ieegTrain);
        ieegTrainFlat = reshape(permute(ieegTrain,[2 1 3]),[matTrain(2) matTrain(1)*matTrain(3)]);
        matTest = size(ieegTest);
        ieegTestFlat = reshape(permute(ieegTest,[2 1 3]),[matTest(2) matTest(1)*matTest(3)]);

        pTrain = labels(train);
        pTest = labels(test);
        uniqLabels = unique(pTrain);
        stmfTemplate = [];
        % Spatio - temporal template
        for iLabel = 1:length(uniqLabels)
            stmfTemplate(iLabel,:) = mean(ieegTrainFlat(pTrain==uniqLabels(iLabel),:),1);
        end
        yhat = []; yscore = [];
        % Estimating correlation between the template & test data
        for iTest = 1:size(ieegTestFlat,1)
            corrTest = [];
            for iLabel = 1:length(uniqLabels)
                corrTest(iLabel) = xcorr(stmfTemplate(iLabel,:)',ieegTestFlat(iTest,:)',0,'coeff');
            end
            yscore(iTest,:) = corrTest;
            [~,maxId] = max(corrTest);
            yhat = [yhat uniqLabels(maxId)];
        end
        aucVect = [];
        % ROC - AUC
        for iLabel = 1:length(uniqLabels)
            [~,~,~,aucVect(iLabel)] = perfcurve(pTest,yscore(:,iLabel),uniqLabels(iLabel));
        end
        
        ytestAll = [ytestAll; pTest'];
        ypredAll = [ypredAll yhat];
        aucAll(nCv,:) = aucVect;
end


end