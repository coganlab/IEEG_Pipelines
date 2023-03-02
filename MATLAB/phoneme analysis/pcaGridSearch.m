function lossVect = pcaGridSearch(gridFeats,labels,varGrid,numFolds)
    % Hyperparameter optimization to extract optimal principal components

    numFeats = length(gridFeats);
    
    % define CV partitions
    cvp = trainTestSplitCV(labels,numFolds);
    lossVect = zeros(cvp.NumTestSets, numel(varGrid{1}));
    for iCV = 1:cvp.NumTestSets
        
        train = cvp.training(iCV);
        test = cvp.test(iCV);

        % get PCA features for each feature type
        trainFeats = cell(1, numFeats);
        testFeats = cell(1, numFeats);
        explVec = cell(1, numFeats);
        for iFeat = 1:numFeats
            gridCurr = gridFeats{iFeat};
            gridTrain = gridCurr(train, :);
            gridTest = gridCurr(test, :);

            [coeffTrain,scoreTrain,~,~,explained] = pca(gridTrain,'Centered',false);
            scoreTest = gridTest*coeffTrain;      
            trainFeats{iFeat} = scoreTrain;
            testFeats{iFeat} = scoreTest;
            explVec{iFeat} = explained; % save amount of variance explained by each feature type
        end
        
        % start grid search
        for iVar = 1:numel(varGrid{1})
            varTrain = [];
            varTest = [];
            % concatenate features with current set of variances explained
            for iFeat = 1:numFeats
                varExpl = varGrid{iFeat}(iVar);
                nModes = find(cumsum(explVec{iFeat})>varExpl,1);
                varTrain = [varTrain trainFeats{iFeat}(:,1:nModes)];
                varTest = [varTest testFeats{iFeat}(:,1:nModes)];
            end

            linearModel = fitcdiscr(varTrain,labels(train),'CrossVal','off','DiscrimType','linear'); 

            % [yhat,yscore] = predict(linearModel, testFeats);
            lossVect(iCV,iVar) = loss(linearModel,varTest,labels(test));
        end
    end
end