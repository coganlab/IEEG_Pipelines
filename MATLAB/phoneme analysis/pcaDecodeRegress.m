function [yhat,lossVect] = pcaDecodeRegress(sigTrain,sigTest,YTrain,YTest,varPercent)

%         meanTrain = mean(sigTrain,1);
%         stdTrain = std(sigTrain,0,1);
        %sigTrainNorm = (sigTrain - meanTrain)./stdTrain;
        [coeffTrain,scoreTrain,~,~,explained] = pca(sigTrain,'Centered',false);
        nModes = find(cumsum(explained)>varPercent,1);
        %sigTestNorm = (sigTest - meanTrain)./stdTrain;
        scoreTest = sigTest*coeffTrain;
        
        scoreTrainGrid = scoreTrain(:,1:nModes);
        scoreTestGrid = scoreTest(:,1:nModes);
%         linearModel = fitglm((scoreTrainGrid),YTrain,'Distribution','binomial'); 
%         yhat = predict(linearModel,scoreTestGrid);
%         lossVect = -mean(YTest.*log(yhat)+(1-YTest).*log(1-yhat));
        linearModel = fitrlinear((scoreTrainGrid),YTrain,'Lambda',1e-3,'Learner','leastsquares'); 
        yhat = predict(linearModel,scoreTestGrid);
        lossVect = loss(linearModel,scoreTestGrid,YTest);
        
        
end