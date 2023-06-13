function [lossVect] = scoreSelectRegress(sig2analyzeAllFeature,labels,varVector,isKfold,numFold)
% Hyperparameter optimization to extract optimal principal components
if(isKfold)
    cvp = cvpartition(length(labels),'HoldOut',1/numFold);
else
    cvp = cvpartition(length(labels),'LeaveOut');
end
   % linearTemplate = templateDiscriminant('DiscrimType','linear');
%    labelCat = [];
%     for iLabel = 1:size(labels,1)
%         labelCat = [labelCat dummyvar(labels(iLabel,:))];
%     end
    
    %[~,scoreall,~,~,~] = pca(sig2analyzeAllFeature,'Centered',false);
       for nCv = 1:cvp.NumTestSets
        
        train = cvp.training(nCv);
        test = cvp.test(nCv);
        featureTrain = sig2analyzeAllFeature(train,:);
        featureTest = sig2analyzeAllFeature(test,:);
        meanTrain = mean(featureTrain,1);
        stdTrain = std(featureTrain,0,1);
        featureTrainNorm = (featureTrain - meanTrain)./stdTrain;
        [coeffTrain,scoreTrain,~,~,explained] = pca(featureTrain,'Centered',false);
       % size(featureTrainNorm)
        featureTestNorm = (featureTest - meanTrain)./stdTrain;
        scoreTest = featureTest*coeffTrain;
        
        for iVar = 1:length(varVector)
            nModes = find(cumsum(explained)>varVector(iVar),1);
            scoreTrainGrid = scoreTrain(:,1:nModes);
            scoreTestGrid = scoreTest(:,1:nModes);
%              size(scoreTrainGrid)
%              size(labels(train))
%            linearModel = fitglm((scoreTrainGrid),labels(train),'Distribution','binomial'); 
            linearModel = fitrlinear((scoreTrainGrid),labels(train),'CrossVal','off','Lambda',1e-3,'Learner','leastsquares'); 
            yhat = predict(linearModel,scoreTestGrid);
            lossVect(nCv,iVar) = loss(linearModel,scoreTestGrid,labels(test));
            
            %lossVect(nCv,iVar) = -mean(labels(test).*log(yhat')+(1-labels(test)).*log(1-yhat'));
%           
        end
    end
end