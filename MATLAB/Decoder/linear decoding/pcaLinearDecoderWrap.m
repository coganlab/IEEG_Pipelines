function [accAll,ytestAll,ypredAll] = pcaLinearDecoderWrap(ieegSplit,labels,tw,etw,numDim,numFolds)
% The function performs supervised PCA-LDA decoding on ephys time-series dataset;
% Step 1: Hyperparameter optimization through nested cross-validation to
% identify the optimal number of PC dimensions
% Step 2: Linear discriminant decoding
%
% Input
% ieegSplit (channels x trials x time): Input dataset
% labels (trials): Input labels
% tw: epoch time-window
% etw: selected time-window for decoding
% numDim: Number of PCA dimensions to include
% numFolds: Number of folds for cross-validation
% Output
% accAll - overall accuracy (0 - 1)
% ytestAll - tested labels
% ypredall - predicted labels

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
        nCv
        train = cvp.training(nCv);
        test = cvp.test(nCv);
        ieegTrain = ieegModel(:,train,:);
        ieegTest = ieegModel(:,test,:);
        matTrain = size(ieegTrain);
        gTrain = reshape(permute(ieegTrain,[2 1 3]),[matTrain(2) matTrain(1)*matTrain(3)]);
        matTest = size(ieegTest);
        gTest = reshape(permute(ieegTest,[2 1 3]),[matTest(2) matTest(1)*matTest(3)]);

        pTrain = labels(train);
        pTest = labels(test);
       
            [lossVect,aucVect] = scoreSelect(gTrain,pTrain,numDim,numFolds); % Hyper parameter tuning
        
        
%          figure;
%          plot(mean(lossVect,1))
        [~,optimDim] = min(mean(lossVect,1)); % Selecting the optimal principal components
%        mean(squeeze(aucVect(:,nDim,:)),1)
        [lossMod,Cmat,yhat,aucVect] = pcaDecode(gTrain,gTest,pTrain,...
                       pTest,optimDim);
    ytestAll = [ytestAll pTest'];
    ypredAll = [ypredAll yhat'];
    accAll = accAll + 1 - lossMod;
    %aucAll = aucAll + aucVect;
    %CmatAll = CmatAll + Cmat;
    end
end