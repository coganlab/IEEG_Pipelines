function [lossVect,aucVect,C] = scoreSelect(sig2analyzeAllFeature,labels,numDim,numFolds)
% Hyperparameter optimization to extract optimal principal components
if(numFolds>0)
    cvp = cvpartition(labels,'KFold',numFolds,'Stratify',true);
else
    cvp = cvpartition(labels,'LeaveOut');
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
        [coeffTrain,scoreTrain] = pca(featureTrainNorm,'Centered',false);
       % size(featureTrainNorm)
        featureTestNorm = (featureTest - meanTrain)./stdTrain;
        scoreTest = featureTestNorm*coeffTrain;
        
        for v = 1:numDim             
            scoreTrainGrid = scoreTrain(:,1:v);
            scoreTestGrid = scoreTest(:,1:v);
%             size(scoreTrainGrid)
%             size(labels(train))
            linearModel = fitcdiscr((scoreTrainGrid),labels(train),'CrossVal','off','DiscrimType','pseudolinear'); 
           %linearModel = fitcknn((scoreTrainGrid),labels(train),'CrossVal','off');
            %linearModel = fitcnb((scoreTrainGrid),labels(train),'CrossVal','off','Prior','uniform');  
            %tempSvm = templateSVM('KernelFunction','rbf');
            %linearModel = fitcecoc((scoreTrainGrid),labels(train),'CrossVal','off','Learners',tempSvm);
            %linearModel = fitcensemble((scoreTrainGrid),labels(train),'CrossVal','off');
            [yhat,yscore] = predict(linearModel,scoreTestGrid);
            lossVect(nCv,v) = loss(linearModel,scoreTestGrid,labels(test));
%             for t = 1:length(labUnique)
%             [~,~,~,aucVect(nCv,v,t)] = perfcurve(labels(test),yscore(:,t),labUnique(t));
%             end
%             C{nCv,v} = confusionmat(labels(test),yhat);
        end
    end
end