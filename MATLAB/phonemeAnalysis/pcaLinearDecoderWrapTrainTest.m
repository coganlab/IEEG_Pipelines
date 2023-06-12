function [accAll,ytestAll,ypredAll,optimVarAll,aucAll, modelWeightsAll] = pcaLinearDecoderWrapTrainTest(ieegSplit,labels,tw,etwTrain,etwTest,varVector,numFolds,isauc)
% The function performs supervised PCA-LDA decoding on ephys time-series dataset;
% Step 1: Hyperparameter optimization through nested cross-validation to
% identify the optimal number of PC dimensions
% Step 2: Linear discriminant decoding
%
% Input
% ieegSplit (channels x trials x time): Input dataset
% labels (trials): Input labels
% tw: epoch time-window e.g [-1 1]
% etwTrain: selected time-window for Training e.g. [-0.5 0]
% etwTest: selected time-window for Testing e.g. [0 0.5]
% varVector: % variance of PCA dimensions to optimize e.g. [10:10:90]
% numFolds: Number of folds for cross-validation e.g. 20 (for 20-fold
% cross-validation)
% isauc: 0|1 to calculate AUC
% Output
% accAll - overall accuracy (0 - 1)
% ytestAll - tested labels
% ypredall - predicted labels

timeSplit = linspace(tw(1),tw(2),size(ieegSplit,3));
timeSelectTrain = timeSplit>=etwTrain(1)&timeSplit<=etwTrain(2);
timeSelectTest = timeSplit>=etwTest(1)&timeSplit<=etwTest(2);
%ieegModel = ieegSplit(:,:,timeSelect);

accAll = 0;
ytestAll = [];
ypredAll = [];
optimVarAll = [];
aucAll = [];
accVectAll = [];
modelWeightsAll = [];
if(numFolds>0)
    cvp = cvpartition(labels,'KFold',numFolds,'Stratify',true);
else
    cvp = cvpartition(labels,'LeaveOut');
end
    for nCv = 1:cvp.NumTestSets
        
        train = cvp.training(nCv);
        test = cvp.test(nCv);
        ieegTrain = ieegSplit(:,train,timeSelectTrain);
        ieegTest = ieegSplit(:,test,timeSelectTest);
        matTrain = size(ieegTrain);
        gTrain = reshape(permute(ieegTrain,[2 1 3]),[matTrain(2) matTrain(1)*matTrain(3)]);
        matTest = size(ieegTest);
        gTest = reshape(permute(ieegTest,[2 1 3]),[matTest(2) matTest(1)*matTest(3)]);

        pTrain = labels(train);
        pTest = labels(test);
        if(length(varVector)>1)
            if(numFolds>0)
                [lossVect] = scoreSelect(gTrain,pTrain,varVector,1,numFolds); % Hyper parameter tuning
            else
                [lossVect] = scoreSelect(gTrain,pTrain,varVector,0,numFolds);
            end

             accVectAll(nCv,:) = mean(lossVect,1);
            [~,optimVarId] = min(mean(lossVect,1)); % Selecting the optimal principal components
            optimVar = varVector(optimVarId);
        else
            optimVar = varVector;
        end
%        mean(squeeze(aucVect(:,nDim,:)),1)
        [lossMod,~,yhat,aucVect,nModes,modelweights] = pcaDecodeVariance(gTrain,gTest,pTrain,...
                       pTest,optimVar,isauc);
    optimVarAll = [optimVarAll optimVar];
%     size(pTest)
     %size(aucVect)
    ytestAll = [ytestAll pTest];
    ypredAll = [ypredAll yhat'];
    accAll = accAll + 1 - lossMod;
    modelWeightsAll{nCv} = modelweights;
    if(isauc)
    aucAll = [aucAll; aucVect];
    end
    %CmatAll = CmatAll + Cmat;
    end
%     figure; 
%     plot(varVector,accVectAll);
    
end