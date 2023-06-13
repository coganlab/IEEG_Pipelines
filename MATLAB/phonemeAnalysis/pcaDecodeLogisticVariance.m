function [lossMod,Cmat,yhat,aucVect,nModes] = pcaDecodeLogisticVariance(sigTrain,sigTest,YTrain,YTest,varPercent, isauc)
        meanTrain = mean(sigTrain,1);
        stdTrain = std(sigTrain,0,1);
        sigTrainNorm = (sigTrain - meanTrain)./stdTrain;
        [coeffTrain,scoreTrain,~,~,explained] = pca(sigTrain,'Centered',false);
        nModes = find(cumsum(explained)>varPercent,1);
        sigTestNorm = (sigTest - meanTrain)./stdTrain;
        scoreTest = sigTest*coeffTrain;
        
        scoreTrainGrid = scoreTrain(:,1:nModes);
        scoreTestGrid = scoreTest(:,1:nModes);
        linearModel = fitclinear((scoreTrainGrid),YTrain,'Learner','logistic'); 
        % linearModel = fitcknn((scoreTrainGrid),YTrain,'CrossVal','off'); 
        %linearModel = fitcnb((scoreTrainGrid),YTrain,'CrossVal','off','Prior','uniform'); 
        %linearModel = fitcensemble((scoreTrainGrid),YTrain,'CrossVal','off');
%          tempSvm = templateSVM('KernelFunction','rbf','Standardize',true);
%             linearModel = fitcecoc((scoreTrainGrid),YTrain,'CrossVal','off','Learners',tempSvm);
        %tempSvm = templateSVM('KernelFunction','linear');
%         tempLogistic = templateKernel('Learner','logistic','KernelScale','auto');
%          linearModel = fitcecoc((scoreTrainGrid),YTrain,'CrossVal','off','Learners',tempLogistic,'Coding','onevsall');
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