function [lossMod,Cmat,yhat,aucVect,nModes,modelweights] = pcaDecodeVariance(sigTrain,sigTest,YTrain,YTest,varPercent, isauc)
%         meanTrain = mean(sigTrain,1);
%         stdTrain = std(sigTrain,0,1);
        %sigTrainNorm = (sigTrain - meanTrain)./stdTrain;
        [coeffTrain,scoreTrain,~,~,explained] = pca(sigTrain,'Centered',false);
        nModes = find(cumsum(explained)>varPercent,1);
        
        %sigTestNorm = (sigTest - meanTrain)./stdTrain;
        scoreTest = sigTest*coeffTrain;
        
        
        scoreTrainGrid = scoreTrain(:,1:nModes);
        scoreTestGrid = scoreTest(:,1:nModes);
        linearModel = fitcdiscr((scoreTrainGrid),YTrain,'CrossVal','off','DiscrimType','linear'); 
        modelweights.pcaScore = coeffTrain(:,1:nModes);
        modelweights.ldamodel = linearModel;
        modelweights.nmodes = nModes;
        
%          tempLinear = templateDiscriminant('DiscrimType','linear');
%          linearModel = fitcensemble((scoreTrainGrid),YTrain,'CrossVal','off','Method','SubSpace','Learners',tempLinear, 'NumLearningCycles',500,'NPredToSample',15);
        %linearModel = fitcknn((scoreTrainGrid),YTrain,'CrossVal','off'); 
        %linearModel = fitcnb((scoreTrainGrid),YTrain,'CrossVal','off','Prior','uniform'); 
        %linearModel = fitcensemble((scoreTrainGrid),YTrain,'CrossVal','off');
%          tempSvm = templateSVM('KernelFunction','rbf','Standardize',true);
%             linearModel = fitcecoc((scoreTrainGrid),YTrain,'CrossVal','off','Learners',tempSvm);
        %tempSvm = templateSVM('KernelFunction','linear');
%          tempLogistic = templateLinear('Learner','svm');
%           linearModel = fitcecoc((scoreTrainGrid),YTrain,'CrossVal','off','Learners',tempLogistic,'Coding','allpairs');
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