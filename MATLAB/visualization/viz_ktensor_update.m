function [elecFact,timeFact,trialFact] = viz_ktensor_update(est_factors,chanMap,selectedChannels,timeTrial,trigLabelsSort,labels,nDisp)
nc = ncomponents(est_factors);
nd = ndims(est_factors);


for iCom = 1:nDisp
    figure;
    elecFact(iCom,:) = est_factors.u{1}(:,iCom);
    timeFact(iCom,:) = est_factors.u{2}(:,iCom);
    trialFact(iCom,:) = est_factors.u{3}(:,iCom);
   subplot(1,5,1);
   %bar(1:length(elecFact),elecFact);
   %title('Electrodes');
   %selectedChannels = sort(chanMap(~isnan(chanMap)));
   chanView(elecFact(iCom,:),chanMap,selectedChannels,isnan(chanMap),'Electrode factors',[],[],[])
   axis equal
   axis tight
   subplot(1,5,2);
   plot(timeTrial,timeFact(iCom,:));
   title('High Gamma Time Factor');
   xlabel('Time (s)');
   axis square
   axis tight
   [trigLabelUnique,~,uniqueId] = unique(trigLabelsSort(:,1));
   subplot(1,5,3);
   plotSpread(trialFact(iCom,:),'distributionIdx',uniqueId','showMM',1);
    set(gca,'xtick',1:length(labels))
     set(gca,'XTickLabel',labels)
   xlabel('Phonemes');
   axis square
   axis tight
   title('Position 1 factors');
   [trigLabelUnique,~,uniqueId] = unique(trigLabelsSort(:,2));
   subplot(1,5,4);
   plotSpread(trialFact(iCom,:),'distributionIdx',uniqueId','showMM',1);
   xlabel('Phonemes');
   title('Position 2 factors');
   set(gca,'xtick',1:length(labels))
     set(gca,'XTickLabel',labels)
   axis square
   axis tight
   [trigLabelUnique,~,uniqueId] = unique(trigLabelsSort(:,3));
   subplot(1,5,5);
   plotSpread(trialFact(iCom,:),'distributionIdx',uniqueId','showMM',1);
   xlabel('Phonemes');
   title('Position 3 factors');
   set(gca,'xtick',1:length(labels))
     set(gca,'XTickLabel',labels)
   axis square
   axis tight
   
%    subplot(1,3,3);
%    jitterAmount = 0.1;
%     jitterValuesX = 2*(rand(size(trigLabelsSort(:,1)))-0.5)*jitterAmount;   % +/-jitterAmount max
%     jitterValuesY = 2*(rand(size(trigLabelsSort(:,2)))-0.5)*jitterAmount;   % +/-jitterAmount max
%    jitterValuesZ = 2*(rand(size(trigLabelsSort(:,3)))-0.5)*jitterAmount; 
%     scatter3(trigLabelsSort(:,1)+jitterValuesX ,trigLabelsSort(:,2)+jitterValuesY,trigLabelsSort(:,3)+jitterValuesZ,40,trialFact(iCom,:),'filled');
%    xlabel('Position 1')
%     ylabel('Position 2')
%     zlabel('Position 3')
%     set(gca,'xtick',1:length(labels))
%     set(gca,'XTickLabel',labels)
%     set(gca,'ytick',1:length(labels))
%     set(gca,'YTickLabel',labels)
%     set(gca,'ztick',1:length(labels))
%     set(gca,'ZTickLabel',labels)
%    cb = colorbar;                                     % create and label the colorbar
%    cb.Label.String = 'Factor score';
   
   
end
end