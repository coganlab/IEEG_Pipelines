function [elecFact,timeFact,trialFact] = viz_ktensor_update_timeSplit(est_factors,chanMap,selectedChannels,timeGammaPerc,timeGammaProd,trigLabelsSort,responseTime,labels,nDisp)



for iCom = 1:nDisp
    
    elecFact(iCom,:) = est_factors.u{1}(:,iCom);
    timeFact(iCom,:) = est_factors.u{2}(:,iCom);
    trialFact(iCom,:) = est_factors.u{3}(:,iCom);
   
  
   
   figure;

   
    t = tiledlayout(3,3,'TileSpacing','compact');
    
    
    
    bgAx = axes(t,'XTick',[],'YTick',[],'Box','off');
    bgAx.Layout.TileSpan = [1 2];


    ax1 = axes(t);
    plot(ax1,timeGammaPerc,timeFact(iCom,1:round(length(timeFact)/2)));
    xline(ax1,timeGammaPerc(end),':');
    ax1.Box = 'off';
    xlim(ax1,[timeGammaPerc(1) timeGammaPerc(end)])
    xlabel(ax1, 'Auditory')

    ax2 = axes(t);
    ax2.Layout.Tile = 2;
    plot(ax2,timeGammaProd,timeFact(iCom,round(length(timeFact)/2)+1:end))
    xline(ax2,timeGammaProd(1),':');
    ax2.YAxis.Visible = 'off';
    ax2.Box = 'off';
    xlim(ax2,[timeGammaProd(1) timeGammaProd(end)])
    xlabel(ax2,'Production')

% Link the axes
    linkaxes([ax1 ax2], 'y')
    
   nexttile([2 1]);
   chanView(elecFact(iCom,:),chanMap,selectedChannels,isnan(chanMap),'Electrode factors',[],[],[]);
   axis equal
   axis tight
   
   [trigLabelUnique,~,uniqueId] = unique(trigLabelsSort(:,1),'sorted');
   
   nexttile;
   plotSpread(trialFact(iCom,:),'distributionIdx',uniqueId','showMM',1);
   set(gca,'xtick',1:length(labels))
    set(gca,'XTickLabel',labels)
   xlabel('Phonemes');
   axis square
   axis tight
   title('Position 1 factors');
   [trigLabelUnique,~,uniqueId] = unique(trigLabelsSort(:,2),'sorted');
   nexttile
   plotSpread(trialFact(iCom,:),'distributionIdx',uniqueId','showMM',1);
   set(gca,'xtick',1:length(labels))
    set(gca,'XTickLabel',labels)
   xlabel('Phonemes');
   title('Position 2 factors');
   axis square
   axis tight
   [trigLabelUnique,~,uniqueId] = unique(trigLabelsSort(:,3),'sorted');
   nexttile
   plotSpread(trialFact(iCom,:),'distributionIdx',uniqueId','showMM',1);
   set(gca,'xtick',1:length(labels))
    set(gca,'XTickLabel',labels)
   xlabel('Phonemes');
   title('Position 3 factors');
   axis square
   axis tight
   
   nexttile;
   [p,tbl] = anova1(trialFact(iCom,:),trigLabelsSort(:,1),'off');
   f1 = tbl{2,5};
   [p,tbl] = anova1(trialFact(iCom,:),trigLabelsSort(:,2),'off');
   f2 = tbl{2,5};
   [p,tbl] = anova1(trialFact(iCom,:),trigLabelsSort(:,3),'off');
   f3 = tbl{2,5};
   scatter(1:3, [f1 f2 f3],'filled');
   hold on;
   line(1:3,[f1 f2 f3]);
   set(gca,'xtick',1:3)
    set(gca,'XTickLabel',{'P1','P2','P3'});
    ylabel('F-statistic');
    title('Phoneme Discrimination');
    axis square
   axis tight
    
   nexttile;
   scatter((trialFact(iCom,:)),responseTime,'filled');
   ylim([0 1.5])
   xlabel('Trial factor');
   ylabel('Response time (s)');
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