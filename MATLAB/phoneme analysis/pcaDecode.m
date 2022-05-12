function [lossMod,Cmat,yhat,aucVect] = pcaDecode(sigTrain,sigTest,YTrain,YTest,nModes,isauc)
        meanTrain = mean(sigTrain,1);
        stdTrain = std(sigTrain,0,1);
        sigTrainNorm = (sigTrain - meanTrain)./stdTrain;
        [coeffTrain,scoreTrain] = pca(sigTrainNorm,'Centered',false);
        
        sigTestNorm = (sigTest - meanTrain)./stdTrain;
        scoreTest = sigTestNorm*coeffTrain;
        
        scoreTrainGrid = scoreTrain(:,1:nModes);
        scoreTestGrid = scoreTest(:,1:nModes);
        linearModel = fitcdiscr((scoreTrainGrid),YTrain,'CrossVal','off','DiscrimType','pseudolinear'); 
        % linearModel = fitcknn((scoreTrainGrid),YTrain,'CrossVal','off'); 
        %linearModel = fitcnb((scoreTrainGrid),YTrain,'CrossVal','off','Prior','uniform'); 
        %linearModel = fitcensemble((scoreTrainGrid),YTrain,'CrossVal','off');
%          tempSvm = templateSVM('KernelFunction','rbf','Standardize',true);
%             linearModel = fitcecoc((scoreTrainGrid),YTrain,'CrossVal','off','Learners',tempSvm);
%         tempSvm = templateSVM('KernelFunction','linear');
%          linearModel = fitcecoc((scoreTrainGrid),YTrain,'CrossVal','off','Learners',tempSvm);
        [yhat,yscore] = predict(linearModel,scoreTestGrid);
        labUnique = unique(YTest);
        aucVect = [];
        if(isauc)
            for t = 1:length(labUnique)
                [~,~,~,aucVect(t)] = perfcurve(YTest,yscore(:,t),labUnique(t));
            end
        end
        lossMod = loss(linearModel,scoreTestGrid,YTest);
        Cmat =  confusionmat(YTest,yhat);
end