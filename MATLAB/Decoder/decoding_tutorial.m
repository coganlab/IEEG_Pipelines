load('ieegFingerDataset.mat');
fsIn = 2000; % input sampling frequency
fsOut = 200; % output sampling frequency
tw = [-2 7]; % epoch time window
etw = [-1.5 6.5]; % output time window to exclude edge artifacts
prtw = [-1.5 -1]; % baseline time window
pstw = [-0.25 0.25]; % response time window

[ieegGammaNorm,~,p_masked] = ExtractHighGammaWrap(ieegSplit,fsIn,fsOut,tw,etw,prtw,pstw);
% ieegGammaNorm - normalized high-gamma
% p_masked - significant channels
[accAll,ytestAll,ypredAll] = pcaLinearDecoderWrap(ieegGammaNorm(p_masked,:,:),fingerLabels,etw,[-0.5 1],60,10);

CmatAll = confusionmat(ytestAll,ypredAll);
acc = trace(CmatAll)/sum(sum(CmatAll))
CmatNorm = CmatAll./sum(CmatAll,2)
figure;
imagesc(CmatNorm.*100);
colormap(jet(4096));
caxis([0 100]);
set(gca,'ytick',1:length(labels));
set(gca,'YTickLabel',labels);
set(gca,'FontSize',20);
set(gca,'xtick',[])
title(['Accuracy : ' num2str(round(acc*100),2) ' %']);