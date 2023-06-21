% Function: pcaDecodeLogisticVariance
% Description: Performs decoding using PCA (Principal Component Analysis) followed by logistic regression on input signals, with a target explained variance percentage.
% Inputs:
%   - sigTrain: Training signals for PCA
%   - sigTest: Test signals for PCA
%   - YTrain: Training labels
%   - YTest: Test labels
%   - varPercent: Target explained variance percentage for selecting the number of PCA modes
%   - isauc: Flag indicating whether to compute AUC
% Outputs:
%   - lossMod: Loss (misclassification rate) for the test data
%   - Cmat: Confusion matrix for the test data
%   - yhat: Predicted labels for the test data
%   - aucVect: AUC values for each class (if isauc is true)
%   - nModes: Number of PCA modes selected

function [lossMod, Cmat, yhat, aucVect, nModes] = pcaDecodeLogisticVariance(sigTrain, sigTest, YTrain, YTest, varPercent, isauc)
    meanTrain = mean(sigTrain, 1); % Compute the mean of the training signals
    stdTrain = std(sigTrain, 0, 1); % Compute the standard deviation of the training signals
    sigTrainNorm = (sigTrain - meanTrain) ./ stdTrain; % Normalize the training signals using mean and standard deviation
    [coeffTrain, scoreTrain, ~, ~, explained] = pca(sigTrain, 'Centered', false); % Perform PCA on the normalized training signals
    
    nModes = find(cumsum(explained) > varPercent, 1); % Select the number of PCA modes based on the target explained variance percentage
    
    sigTestNorm = (sigTest - meanTrain) ./ stdTrain; % Normalize the test signals using mean and standard deviation
    scoreTest = sigTest * coeffTrain; % Apply the PCA transformation to the test signals
    
    scoreTrainGrid = scoreTrain(:, 1:nModes); % Select the desired number of PCA modes for training
    scoreTestGrid = scoreTest(:, 1:nModes); % Select the corresponding PCA modes for test
    
    linearModel = fitclinear(scoreTrainGrid, YTrain, 'Learner', 'logistic'); % Train a logistic regression model on the selected PCA modes
    
    [yhat, yscore] = predict(linearModel, scoreTestGrid); % Predict labels for the test data using the trained logistic regression model
    
    labUnique = unique(YTest); % Get the unique labels in the test data
    aucVect = []; % Initialize the vector to store AUC values
    if (isauc) % If AUC computation is enabled
        for t = 1:length(labUnique) % Loop through each unique label
            [~, ~, ~, aucVect(t)] = perfcurve(YTest, yscore(:, t), labUnique(t)); % Compute AUC for each class
        end
    end
    
    lossMod = loss(linearModel, scoreTestGrid, YTest); % Compute the loss (misclassification rate) for the test data
    Cmat = confusionmat(YTest, yhat); % Compute the confusion matrix for the test data
end
