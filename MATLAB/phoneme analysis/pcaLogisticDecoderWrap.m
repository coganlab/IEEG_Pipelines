function [accAll, ytestAll, ypredAll, optimVarAll, aucAll] = pcaLogisticDecoderWrap(ieegSplit, labels, tw, etw, varVector, numFolds, isauc)
% This function performs supervised PCA-LDA decoding on an ephys time-series dataset.
% Step 1: Hyperparameter optimization through nested cross-validation to identify the optimal number of PC dimensions.
% Step 2: Logistic discriminant decoding.
%
% Inputs:
%   - ieegSplit: Input dataset of size (channels x trials x time)
%   - labels: Input labels for each trial
%   - tw: Epoch time-window
%   - etw: Selected time-window for decoding
%   - varVector: Variance of PCA dimensions to optimize
%   - numFolds: Number of folds for cross-validation
%   - isauc: Flag indicating whether to calculate AUC
%
% Outputs:
%   - accAll: Overall accuracy (0-1)
%   - ytestAll: Tested labels
%   - ypredAll: Predicted labels
%   - optimVarAll: Optimal number of principal components
%   - aucAll: Area under the ROC curve (if isauc=true)

timeSplit = linspace(tw(1), tw(2), size(ieegSplit, 3));
timeSelect = timeSplit >= etw(1) & timeSplit <= etw(2);
ieegModel = ieegSplit(:, :, timeSelect);

accAll = 0;
ytestAll = [];
ypredAll = [];
optimVarAll = [];
aucAll = [];
accVectAll = [];

if (numFolds > 0)
    cvp = cvpartition(labels, 'KFold', numFolds, 'Stratify', true);
else
    cvp = cvpartition(labels, 'LeaveOut');
end

for nCv = 1:cvp.NumTestSets
    train = cvp.training(nCv);
    test = cvp.test(nCv);
    ieegTrain = ieegModel(:, train, :);
    ieegTest = ieegModel(:, test, :);
    matTrain = size(ieegTrain);
    gTrain = reshape(permute(ieegTrain, [2 1 3]), [matTrain(2) matTrain(1)*matTrain(3)]);
    matTest = size(ieegTest);
    gTest = reshape(permute(ieegTest, [2 1 3]), [matTest(2) matTest(1)*matTest(3)]);

    pTrain = labels(train);
    pTest = labels(test);
    
    if (length(varVector) > 1)
        if (numFolds > 0)
            [lossVect] = scoreSelectLogistic(gTrain, pTrain, varVector, 1, numFolds); % Hyperparameter tuning
        else
            [lossVect] = scoreSelectLogistic(gTrain, pTrain, varVector, 0, numFolds);
        end

        accVectAll(nCv, :) = mean(lossVect, 1);
        [~, optimVarId] = min(mean(lossVect, 1)); % Selecting the optimal principal components
        optimVar = varVector(optimVarId);
    else
        optimVar = varVector;
    end
    
    [lossMod, Cmat, yhat, aucVect] = pcaDecodeLogisticVariance(gTrain, gTest, pTrain, pTest, optimVar, isauc);
    
    optimVarAll = [optimVarAll optimVar];
    ytestAll = [ytestAll pTest];
    ypredAll = [ypredAll yhat'];
    accAll = accAll + 1 - lossMod;
    
    if (isauc)
        aucAll = [aucAll; aucVect];
    end
end

end
