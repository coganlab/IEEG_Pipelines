function [accAll,ytestAll,ypredAll,aucAll] = stmfDecodeWrap(ieegSplit,labels,tw,etw,numFolds, isauc)
timeSplit = linspace(tw(1),tw(2),size(ieegSplit,3));
timeSelect = timeSplit>=etw(1)&timeSplit<=etw(2);
ieegModel = ieegSplit(:,:,timeSelect);

accAll = 0;

ytestAll = [];
ypredAll = [];

if(numFolds>0)
cvp = cvpartition(labels,'KFold',numFolds);
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
        uniqLabels = unique(pTrain);
        stmfTemplate = [];
        for iLabel = 1:length(uniqLabels)
            stmfTemplate(iLabel,:) = mean(gTrain(pTrain==uniqLabels(iLabel),:),1);
        end
        yhat = []; yscore = [];
        for iTest = 1:size(gTest,1)
            corrTest = [];
            for iLabel = 1:length(uniqLabels)
                corrTest(iLabel) = xcorr(stmfTemplate(iLabel,:)',gTest(iTest,:)',0,'coeff');
            end
            yscore(iTest,:) = corrTest;
            [~,maxId] = max(corrTest);
            yhat = [yhat uniqLabels(maxId)];
        end
        if(isauc)
        aucVect = [];
        for iLabel = 1:length(uniqLabels)
            [~,~,~,aucVect(iLabel)] = perfcurve(pTest,yscore(:,iLabel),uniqLabels(iLabel));
        end
        aucAll(nCv,:) = aucVect;
        end
        
        ytestAll = [ytestAll; pTest'];
        ypredAll = [ypredAll yhat];
        
end


end