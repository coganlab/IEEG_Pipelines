
% Function: pcaDecodeRegress
% Description: Performs decoding using PCA (Principal Component Analysis) followed by linear regression on input signals.
% Inputs:
%   - sigTrain: Training signals for PCA
%   - sigTest: Test signals for PCA
%   - YTrain: Training labels
%   - YTest: Test labels
%   - varPercent: Target explained variance percentage for selecting the number of PCA modes
% Outputs:
%   - yhat: Predicted labels for the test data
%   - lossVect: Loss vector for the test data

function [yhat, lossVect] = pcaDecodeRegress(sigTrain, sigTest, YTrain, YTest, varPercent)
%         meanTrain = mean(sigTrain,1);
%         stdTrain = std(sigTrain,0,1);
        %sigTrainNorm = (sigTrain - meanTrain)./stdTrain;
        [coeffTrain, scoreTrain, ~, ~, explained] = pca(sigTrain, 'Centered', false); % Perform PCA on the training signals
        
        nModes = find(cumsum(explained) > varPercent, 1); % Select the number of PCA modes based on the target explained variance percentage
        
        %sigTestNorm = (sigTest - meanTrain)./stdTrain;
        scoreTest = sigTest * coeffTrain; % Apply the PCA transformation to the test signals
        
        scoreTrainGrid = scoreTrain(:, 1:nModes); % Select the desired number of PCA modes for training
        scoreTestGrid = scoreTest(:, 1:nModes); % Select the corresponding PCA modes for test
        
%         linearModel = fitglm(scoreTrainGrid, YTrain, 'Distribution', 'binomial'); 
%         yhat = predict(linearModel, scoreTestGrid);
%         lossVect = -mean(YTest .* log(yhat) + (1 - YTest) .* log(1 - yhat));

        linearModel = fitrlinear(scoreTrainGrid, YTrain, 'Lambda', 1e-3, 'Learner', 'leastsquares'); % Train a linear regression model on the selected PCA modes
        yhat = predict(linearModel, scoreTestGrid); % Predict labels for the test data using the trained linear regression model
        lossVect = loss(linearModel, scoreTestGrid, YTest); % Compute the loss vector for the test data
end

