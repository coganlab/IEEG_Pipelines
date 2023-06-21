% Function: linearDecoder
% Description: Performs linear decoding using iEEG data and labels.
% Inputs:
%   - ieegSplit: iEEG data split into trials (channels x trials x time)
%   - labels: Labels for each trial
%   - tw: Time window 
%   - etw: Decoding time window
%   - numFolds: Number of folds for cross-validation (0 for leave-one-out)
%   - isauc: Flag to compute area under the curve (AUC)
% Outputs:
%   - accAll: Accuracy for each cross-validation fold
%   - ytestAll: True labels for each cross-validation fold
%   - ypredAll: Predicted labels for each cross-validation fold
%   - aucAll: AUC values for each class (if isauc is true)

function [accAll, ytestAll, ypredAll, aucAll] = linearDecoder(ieegSplit, labels, tw, etw, numFolds, isauc)
    % Compute time split based on tw (time window)
    timeSplit = linspace(tw(1), tw(2), size(ieegSplit, 3));
    % Select time points within etw (decodting time window)
    timeSelect = timeSplit >= etw(1) & timeSplit <= etw(2);
    % Select iEEG data for the decoding time window
    ieegModel = ieegSplit(:, :, timeSelect);
    
    % Initialize variables for storing results
    accAll = 0;
    aucAll = [];
    ytestAll = [];
    ypredAll = [];
    
    % Create cross-validation partition based on the number of folds
    if (numFolds > 0)
        cvp = cvpartition(labels, 'KFold', numFolds, 'Stratify', true);
    else
        cvp = cvpartition(labels, 'LeaveOut');
    end
    
    % Iterate through each cross-validation fold
    for nCv = 1:cvp.NumTestSets
        train = cvp.training(nCv); % Indices for training set
        test = cvp.test(nCv); % Indices for test set
        
        ieegTrain = ieegModel(:, train, :); % Select iEEG data for training set
        ieegTest = ieegModel(:, test, :); % Select iEEG data for test set
        
        matTrain = size(ieegTrain);
        gTrain = reshape(permute(ieegTrain, [2 1 3]), [matTrain(2), matTrain(1) * matTrain(3)]);
        
        matTest = size(ieegTest);
        gTest = reshape(permute(ieegTest, [2 1 3]), [matTest(2), matTest(1) * matTest(3)]);
        
        pTrain = labels(train); % Labels for training set
        pTest = labels(test); % Labels for test set
        
        meanTrain = mean(gTrain, 1); % Compute mean of the training data
        stdTrain = std(gTrain, 0, 1); % Compute standard deviation of the training data
        
        % Normalize the training and test data based on mean and standard deviation
        gTrainNorm = (gTrain - meanTrain) ./ stdTrain;
        gTestNorm = (gTest - meanTrain) ./ stdTrain;
        
        % Fit the decoding model using the training data
        tempLogistic = templateKernel('Learner', 'logistic', 'KernelScale', 'auto');
        phonemeDecodeModel = fitcecoc(gTrainNorm, pTrain, 'CrossVal', 'off', 'Learners', tempLogistic, 'Coding', 'onevsall');
      
        % Predict labels for the test data
        [yhat, yscore] = predict(phonemeDecodeModel, gTestNorm);
        
        % Compute AUC for each class (if isauc is true)
        labUnique = unique(pTest);
        aucVect = [];
        for t = 1:length(labUnique)
            [~, ~, ~, aucVect(t)] = perfcurve(pTest, yscore(:, t), labUnique(t));
        end
        
        % Compute loss (misclassification rate) for the test data
        lossMod = loss(phonemeDecodeModel, gTest, pTest);
        
        % Append true labels and predicted labels to the overall results
        ytestAll = [ytestAll pTest];
        ypredAll = [ypredAll yhat'];
        
        % Compute accuracy for the current fold and add it to the overall accuracy
        accAll = accAll + 1 - lossMod;
        
        % Append AUC values for each class (if isauc is true)
        if (isauc)
            aucAll = [aucAll; aucVect];
        end
    end
end
