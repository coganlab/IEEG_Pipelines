% Function: pcaDecodeVariance
% Description: Performs decoding using PCA (Principal Component Analysis) followed by linear discriminant analysis on input signals.
% Inputs:
%   - sigTrain: Training signals for PCA
%   - sigTest: Test signals for PCA
%   - YTrain: Training labels
%   - YTest: Test labels
%   - varPercent: Target explained variance percentage for selecting the number of PCA modes
%   - isauc: Boolean flag indicating whether to compute AUC (Area Under the Curve) values
% Outputs:
%   - lossMod: Loss of the trained model on the test data
%   - Cmat: Confusion matrix of the predicted labels on the test data
%   - yhat: Predicted labels for the test data
%   - aucVect: AUC values for each class (if isauc is true)
%   - nModes: Number of selected PCA modes
%   - modelweights: Structure containing the PCA score, LDA model, and number of PCA modes

function [lossMod, Cmat, yhat, aucVect, nModes, modelweights] = pcaDecodeVariance(sigTrain, sigTest, YTrain, YTest, varPercent, isauc)
%         meanTrain = mean(sigTrain,1);
%         stdTrain = std(sigTrain,0,1);
        %sigTrainNorm = (sigTrain - meanTrain)./stdTrain;
        [coeffTrain, scoreTrain, ~, ~, explained] = pca(sigTrain, 'Centered', false); % Perform PCA on the training signals
        nModes = find(cumsum(explained) > varPercent, 1); % Select the number of PCA modes based on the target explained variance percentage
        
        %sigTestNorm = (sigTest - meanTrain)./stdTrain;
        scoreTest = sigTest * coeffTrain; % Apply the PCA transformation to the test signals
        
        scoreTrainGrid = scoreTrain(:, 1:nModes); % Select the desired number of PCA modes for training
        scoreTestGrid = scoreTest(:, 1:nModes); % Select the corresponding PCA modes for test
        
        linearModel = fitcdiscr(scoreTrainGrid, YTrain, 'CrossVal', 'off', 'DiscrimType', 'pseudolinear'); % Train a linear discriminant analysis (LDA) model on the selected PCA modes
        modelweights.pcaScore = coeffTrain(:, 1:nModes); % Store the PCA scores for the selected modes
        modelweights.ldamodel = linearModel; % Store the trained LDA model
        modelweights.nmodes = nModes; % Store the number of selected PCA modes
        
        [yhat, yscore] = predict(linearModel, scoreTestGrid); % Predict labels for the test data using the trained LDA model
        labUnique = unique(YTest);
        aucVect = [];
        if (isauc)
            for t = 1:length(labUnique)
                [~, ~, ~, aucVect(t)] = perfcurve(YTest, yscore(:, t), labUnique(t)); % Compute AUC values for each class
            end
        end
        lossMod = loss(linearModel, scoreTestGrid, YTest); % Compute the loss of the trained model on the test data
        Cmat = confusionmat(YTest, yhat); % Compute the confusion matrix of the predicted labels on the test data
end
