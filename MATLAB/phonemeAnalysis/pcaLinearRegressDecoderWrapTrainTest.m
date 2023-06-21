function [ytestAll, ypredAll, optimDimAll] = pcaLinearRegressDecoderWrapTrainTest(ieegSplit, predictors, tw, etwTrain, etwTest, varVector, numFolds)
% This function performs PCA-LDA decoding on EEG time-series data using linear regression.
%
% Input:
% ieegSplit (channels x trials x time): EEG data split into training and testing sets
% predictors (trials): Predictors/labels corresponding to the EEG data
% tw: epoch time-window e.g. [-1 1]
% etwTrain: selected time-window for Training e.g. [-0.5 0]
% etwTest: selected time-window for Testing e.g. [0 0.5]
% varVector: variance of PCA dimensions to optimize e.g. [10:10:90]
% numFolds: Number of folds for cross-validation e.g. 20 (for 20-fold cross-validation)
%
% Output:
% ytestAll: True predictors of the testing set
% ypredAll: Predicted predictors of the testing set
% optimDimAll: Optimal dimensions for PCA

timeSplit = linspace(tw(1), tw(2), size(ieegSplit, 3));
timeSelectTrain = timeSplit >= etwTrain(1) & timeSplit <= etwTrain(2);
timeSelectTest = timeSplit >= etwTest(1) & timeSplit <= etwTest(2);

lossAll = 0;
ytestAll = [];
ypredAll = [];
optimDimAll = [];
accVectAll = [];

if (numFolds > 0)
    cvp = cvpartition(length(predictors), 'KFold', numFolds);
else
    cvp = cvpartition(length(predictors), 'LeaveOut');
end

for nCv = 1:cvp.NumTestSets
    train = cvp.training(nCv);
    test = cvp.test(nCv);
    ieegTrain = ieegSplit(:, train, timeSelectTrain);
    ieegTest = ieegSplit(:, test, timeSelectTest);
    matTrain = size(ieegTrain);
    gTrain = reshape(permute(ieegTrain, [2 1 3]), [matTrain(2) matTrain(1) * matTrain(3)]);
    matTest = size(ieegTest);
    gTest = reshape(permute(ieegTest, [2 1 3]), [matTest(2) matTest(1) * matTest(3)]);

    pTrain = predictors(train);
    pTest = predictors(test);

    if (length(varVector) > 1)
        if (numFolds > 0)
            [lossVect] = scoreSelectRegress(gTrain, pTrain, varVector, 1, numFolds); % Hyperparameter tuning
        else
            [lossVect] = scoreSelectRegress(gTrain, pTrain, varVector, 0, numFolds);
        end

        accVectAll(nCv, :) = mean(lossVect, 1);
        [~, optimVarId] = min(mean(lossVect, 1)); % Selecting the optimal principal components
        optimVar = varVector(optimVarId);
    else
        optimVar = varVector;
    end

    [yhat, loss] = pcaDecodeRegress(gTrain, gTest, pTrain, pTest, optimVar);

    optimDimAll = [optimDimAll optimVar];
    ytestAll = [ytestAll pTest];
    ypredAll = [ypredAll yhat'];
    lossAll = [lossAll loss];
end

end
