function decodeResultStruct = pcaLinearDecoderWrapMultiFeature(ieegFeats,labels,tw,etw,sigChannel,varExpl,numFolds,nIter,isauc)
    % The function performs supervised PCA-LDA decoding on ephys time-series dataset;
    % Step 1: Hyperparameter optimization through nested cross-validation to
    % identify the optimal number of PC dimensions
    % Step 2: Linear discriminant decoding
    %
    % Input
    % ieegFeats (1 x numFeats cell {channels x trials x time}): Input dataset 
    %       split by feature type (HG, LFS, etc.)
    % labels (trials): Input labels for each trial
    % tw: epoch time-window
    % etw: selected time-window for decoding
    % sigChannel: significant channels to use for decoding
    % varExpl (1 x numFeats cell {double value or vector}): Amount of variance
    %       explained by each feature for PCA decomposition or list of variance
    %       explained values to perform grid search over   
    % numFolds: Number of folds for cross-validation
    % nIter: Number of iterations for model evaluation
    % isauc: Boolean to compute AUC
    %
    % Output
    % accAll - overall accuracy (0 - 1)
    % ytestAll - tested labels
    % ypredall - predicted labels


    numFeats = length(ieegFeats);
    timeSplit = linspace(tw(1),tw(2),size(ieegFeats{1},3));
    timeSelect = timeSplit>=etw(1)&timeSplit<=etw(2);

    varLens = cellfun(@length, varExpl);

    accAll = 0;
    yTestAll = [];
    yPredAll = [];
    aucAll = [];
    optimVarHist = [];  

    uniqLabels = unique(labels);
    CmatAll = zeros(length(uniqLabels),length(uniqLabels));

    % 1. Outer for-loop for number CV iterations

    for iter = 1:nIter

        % 2. Partition train and test data into CV folds
        cvp = trainTestSplitCV(labels,numFolds);

        yTestCV = [];
        yPredCV = [];

        % 3. Iterate through CV folds
        for iCV = 1:cvp.NumTestSets
 
            train = cvp.training(iCV);
            test = cvp.test(iCV);
            
            % Partition features into train and test for fold
            trainFeats = cell(1, numFeats);
            testFeats = cell(1, numFeats);
            for iFeat = 1:numFeats

                ieegCurr = ieegFeats{iFeat};
                ieegModel = ieegCurr(sigChannel,:,timeSelect);
                ieegTrain = ieegModel(:,train,:);
                ieegTest = ieegModel(:,test,:);
                % reshape to trials x time*channels (i.e. observations x features)
                matTrain = size(ieegTrain);
                trainFeats{iFeat} = reshape(permute(ieegTrain,[2 1 3]),[matTrain(2) matTrain(1)*matTrain(3)]);
                
                matTest = size(ieegTest);
                testFeats{iFeat} = reshape(permute(ieegTest,[2 1 3]),[matTest(2) matTest(1)*matTest(3)]);
            end
            yTrain = labels(train);
            yTest = labels(test);

            % Cross-validation on amount of variance explained for each feature
            if max(varLens) > 1
                disp('Performing grid search for optimal number of principal components')
                disp(iCV)
                optimFeatVars = zeros(1,numFeats);
                varGrid = cell(1, numFeats);
                [varGrid{:}] = ndgrid(varExpl{:});

                lossVect = pcaGridSearch(trainFeats, yTrain, varGrid, numFolds);
                [~,optimVarId] = min(mean(lossVect,1)); % mean loss across grid search folds
                for iFeat = 1:numFeats
                    if length(varExpl{iFeat}) > 1  % use grid search selected value
                        optimFeatVars(iFeat) = varGrid{iFeat}(optimVarId);
                    else  % use provided value (if optimization is performed for some features but not others)
                        optimFeatVars(iFeat) = varExpl{iFeat};
                    end
                    % optimFeatVars(iFeat) = varGrid{iFeat}(optimVarId);
                end
                disp(optimFeatVars)
            else
                disp('No PC optimization')
                disp(iCV)
                optimFeatVars = [varExpl{:}];
            end
            % save optimal variance explained history
            optimVarHist = [optimVarHist; optimFeatVars];

            pcaTrainFeats = [];
            pcaTestFeats = [];
            for iFeat = 1:numFeats
                % 5. Concatenate PCA transformed features (Loop from 4 done)

                % PCA transform current feature and add to feature collection
                [scoreTrain, scoreTest, ~] = pcaTransform(trainFeats{iFeat}, testFeats{iFeat}, optimFeatVars(iFeat));
                pcaTrainFeats = [pcaTrainFeats scoreTrain];
                pcaTestFeats = [pcaTestFeats scoreTest];
            end
    

            % 7. Input concatenated features to LDA decoder

            [lossMod, ~, yHat, aucVect] = linDscrDecode(pcaTrainFeats, yTrain, pcaTestFeats, yTest, isauc);

            % 8. Collect performance metrics across CV (end loops from 3 and 4)

            yTestCV = [yTestCV yTest];
            yPredCV = [yPredCV yHat'];
            accAll = accAll + 1 - lossMod;

            if(isauc)
                aucAll = [aucAll; aucVect];
            end
        end

        Cmat = confusionmat(yTestCV,yPredCV);
        CmatAll = CmatAll + Cmat;

        yTestAll = [yTestAll yTestCV];
        yPredAll = [yPredAll yPredCV];

    end

    % 9. Collect performance metrics across CV iterations (end loop from 1)

    CmatCatNorm = CmatAll./sum(CmatAll,2);
      
    decodeResultStruct.accPhoneme = trace(CmatCatNorm)/size(CmatCatNorm,1); 
    decodeResultStruct.accPhonemeUnBias = trace(CmatAll)/sum(CmatAll(:));
    decodeResultStruct.cmat = CmatCatNorm;
    decodeResultStruct.p = StatThInv(yTestAll, decodeResultStruct.accPhoneme.*100);
    decodeResultStruct.optimVarHist = optimVarHist;  
end
  