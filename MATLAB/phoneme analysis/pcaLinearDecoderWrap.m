function [accAll, ytestAll, ypredAll, optimVarAll, aucAll, modelWeightsAll] = pcaLinearDecoderWrap(ieegSplit, labels, tw, etw, varVector, numFolds, isauc)
    % Performs supervised PCA-LDA decoding on ephys time-series dataset.
    % Step 1: Hyperparameter optimization through nested cross-validation to
    % identify the optimal number of PC dimensions.
    % Step 2: Linear discriminant decoding.
    %
    % Input:
    % ieegSplit (channels x trials x time): Input dataset.
    % labels (trials): Input labels.
    % tw: Epoch time-window.
    % etw: Selected time-window for decoding.
    % varVector: Number of PCA dimensions to include or an array of values for hyperparameter tuning.
    % numFolds: Number of folds for cross-validation.
    % isauc: Flag indicating whether to compute AUC (true/false).
    %
    % Output:
    % accAll: Overall accuracy (0 - 1).
    % ytestAll: Tested labels.
    % ypredAll: Predicted labels.
    % optimVarAll: Optimal number of PC dimensions for each fold.
    % aucAll: AUC values for each fold (if isauc is true).
    % modelWeightsAll: Model weights for each fold.

    % Generate time split based on tw
    timeSplit = linspace(tw(1), tw(2), size(ieegSplit, 3));

    % Select time indices within the specified etw
    timeSelect = timeSplit >= etw(1) & timeSplit <= etw(2);

    % Extract the selected time window from the input dataset
    ieegModel = ieegSplit(:, :, timeSelect);

    % Initialize variables
    accAll = 0;
    ytestAll = [];
    ypredAll = [];
    optimVarAll = [];
    aucAll = [];
    lossVectAll = [];
    modelWeightsAll = [];

    % Perform cross-validation
    if (numFolds > 0)
        cvp = cvpartition(labels, 'KFold', numFolds, 'Stratify', true);
    else
        cvp = cvpartition(labels, 'LeaveOut');
    end

    for nCv = 1:cvp.NumTestSets
        % Split the data into training and testing sets
        train = cvp.training(nCv);
        test = cvp.test(nCv);
        ieegTrain = ieegModel(:, train, :);
        ieegTest = ieegModel(:, test, :);
        matTrain = size(ieegTrain);
        xTrain = reshape(permute(ieegTrain, [2 1 3]), [matTrain(2) matTrain(1) * matTrain(3)]);
        matTest = size(ieegTest);
        xTest = reshape(permute(ieegTest, [2 1 3]), [matTest(2) matTest(1) * matTest(3)]);

        yTrain = labels(train);
        yTest = labels(test);

        if (length(varVector) > 1)
            if (numFolds > 0)
                % Perform hyperparameter tuning
                [lossVect] = scoreSelect(xTrain, yTrain, varVector, 1, numFolds);
            else
                [lossVect] = scoreSelect(xTrain, yTrain, varVector, 0, numFolds);
            end

            lossVectAll(nCv, :) = mean(lossVect, 1);

            % Select the optimal principal components based on the mean loss
            [~, optimVarId] = min(mean(lossVect, 1));
            optimVar = varVector(optimVarId);
        else
            optimVar = varVector;
        end

        % Perform PCA-LDA decoding
        [lossMod, Cmat, yhat, aucVect, nModes, modelweights] = pcaDecodeVariance(xTrain, xTest, yTrain, yTest, optimVar, isauc);
        
        % Store the results for this fold
        optimVarAll = [optimVarAll optimVar];
        ytestAll = [ytestAll yTest];
        ypredAll = [ypredAll yhat'];
        accAll = accAll + 1 - lossMod;
        modelweights.optimVar = optimVarAll;
        modelWeightsAll{nCv} = modelweights;

        % Store the AUC values if isauc is true
        if (isauc)
            aucAll = [aucAll; aucVect];
        end
    end
end
