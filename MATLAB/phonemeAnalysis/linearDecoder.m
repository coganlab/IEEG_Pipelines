function [accAll,ytestAll,ypredAll,aucAll] = linearDecoder(ieegSplit,labels,tw,etw,numFolds, isauc)
timeSplit = linspace(tw(1),tw(2),size(ieegSplit,3));
timeSelect = timeSplit>=etw(1)&timeSplit<=etw(2);
ieegModel = ieegSplit(:,:,timeSelect);

accAll = 0;
aucAll = [];
ytestAll = [];
ypredAll = [];

if(numFolds>0)
cvp = cvpartition(labels,'KFold',numFolds,'Stratify',true);
else
    cvp = cvpartition(labels,'LeaveOut');
end


for nCv = 1:cvp.NumTestSets
        
        train = cvp.training(nCv);
        test = cvp.test(nCv);
        ieegTrain = ieegModel(:,train,:);
        ieegTest = ieegModel(:,test,:);
        matTrain = size(ieegTrain);
        gTrain = reshape(permute(ieegTrain,[2 1 3]),[matTrain(2) matTrain(1)*matTrain(3)]);
        matTest = size(ieegTest);
        gTest = reshape(permute(ieegTest,[2 1 3]),[matTest(2) matTest(1)*matTest(3)]);

        pTrain = labels(train);
        pTest = labels(test);
        meanTrain = mean(gTrain,1);
        stdTrain = std(gTrain,0,1);
        gTrainNorm = (gTrain - meanTrain)./stdTrain;
        gTestNorm = (gTest - meanTrain)./stdTrain;
        size(gTrain)
        %phonemeDecodeModel = fitcdiscr(gTrainNorm,pTrain,'CrossVal','off','DiscrimType','pseudolinear');
        tempLogistic = templateKernel('Learner','logistic','KernelScale','auto');
        phonemeDecodeModel = fitcecoc((gTrainNorm),pTrain,'CrossVal','off','Learners',tempLogistic,'Coding','onevsall');
      
        [yhat,yscore] = predict(phonemeDecodeModel,gTestNorm);
        labUnique = unique(pTest);
        aucVect = [];
        for t = 1:length(labUnique)
            [~,~,~,aucVect(t)] = perfcurve(pTest,yscore(:,t),labUnique(t));
        end
        lossMod = loss(phonemeDecodeModel,gTest,pTest);
        ytestAll = [ytestAll pTest];
        ypredAll = [ypredAll yhat'];
        accAll = accAll + 1 - lossMod;
        if(isauc)
            aucAll = [aucAll; aucVect];
        end
end