function [scoreTrainGrid,scoreTestGrid,nModes] = pcaTransform(sigTrain,sigTest,varPercent)

    [coeffTrain,scoreTrain,~,~,explained] = pca(sigTrain,'Centered',false);
    nModes = find(cumsum(explained)>varPercent,1);
    scoreTest = sigTest*coeffTrain;
    
    scoreTrainGrid = scoreTrain(:,1:nModes);
    scoreTestGrid = scoreTest(:,1:nModes);  
end