function [ypred] = nbClassifyPSCA(sigPower,fs,labels,pcaDim,isNorm)
freq=[0:size(sigPower,3)-1].*fs/2./(size(sigPower,3)-1);
      % psd_all=log(psd_all(:,freq>5&freq<200)./(mean(psd_all(:,freq>5&freq<200))));
for iTest = 1:length(labels)
    testId = iTest;
    trainIdPsd = setdiff(1:size(sigPower,2),iTest);
    trainId = setdiff(1:length(labels),iTest);
    YTrain = labels(trainId);
    YTest = labels(testId);
    for iChan = 1:size(sigPower,1)
        psdChan = squeeze(sigPower(iChan,:,freq>=5&freq<=200));
        psdTrain = psdChan(trainIdPsd,:);
        psdTest = psdChan(testId,:);
        psdTrainNorm = log(psdTrain./mean(psdTrain,1));
        psdTestNorm = log(psdTest./mean(psdTrain,1));
%         figure;
%         plot(psdTestNorm);
        [psdCoef,psdScore] = pca(psdTrainNorm,'Centered',false);
        scoreTrain(:,iChan) = psdScore(1:length(trainId),pcaDim);
        scoreTest(:,iChan) = psdTestNorm*psdCoef(:,pcaDim);
    end
    if(isNorm)
    mTrain = mean(scoreTrain,1);
    sTrain = std(scoreTrain,0,1);
    XTrainNorm = (scoreTrain-mTrain)./sTrain;
    XTestNorm = (scoreTest - mTrain)./sTrain;
    else
        XTrainNorm = scoreTrain;
        XTestNorm = scoreTest;
    end
    fingerMdl=fitcnb(XTrainNorm,YTrain,'prior','uniform');
    ypred(iTest) = predict(fingerMdl,XTestNorm);
end
end