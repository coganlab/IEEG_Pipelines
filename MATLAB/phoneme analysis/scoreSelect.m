function [lossVect,aucVect,phonErrVect] = scoreSelect(sig2analyzeAllFeature,labels,varVector,isKfold,numFold)
% Hyperparameter optimization to extract optimal principal components
if(isKfold)
    cvp = cvpartition(labels,'KFold',numFold,'Stratify',true);
else
    cvp = cvpartition(labels,'KFold',20,'Stratify',true);
end
   % linearTemplate = templateDiscriminant('DiscrimType','linear');
    labUnique = unique(labels);
    aucVect = []; scoreF = [];
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
           linearModel = fitcdiscr((scoreTrainGrid),labels(train),'CrossVal','off','DiscrimType','linear'); 
           %linearModel = fitcknn((scoreTrainGrid),labels(train),'CrossVal','off');
            %linearModel = fitcnb((scoreTrainGrid),labels(train),'CrossVal','off','Prior','uniform');  
%              tempLogistic = templateKernel('Learner','logistic','KernelScale','auto');
%             linearModel = fitcecoc((scoreTrainGrid),labels(train),'CrossVal','off','Learners',tempLogistic,'Coding','onevsall');
            %linearModel = fitcensemble((scoreTrainGrid),labels(train),'CrossVal','off');
            [yhat,yscore] = predict(linearModel,scoreTestGrid);
            lossVect(nCv,iVar) = loss(linearModel,scoreTestGrid,labels(test));
            cmat = confusionmat(labels(test),yhat');
            cmatNorm = cmat./sum(cmat,2);
%             size(cmatNorm)
%             unique(labels(test))
            %phonErrVect(nCv,iVar) = phonemeDistanceError(cmatNorm,unique([labels(test) yhat']));
%             accVect(nCv,iVar) =  trace(cmatNorm)./size(cmatNorm,1);
%             for t = 1:length(labUnique)
%             [~,~,~,aucVect(nCv,v,t)] = perfcurve(labels(test),yscore(:,t),labUnique(t));
%             end
%             C{nCv,v} = confusionmat(labels(test),yhat);
        end
    end
end