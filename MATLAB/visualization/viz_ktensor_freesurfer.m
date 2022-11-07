function [elecFact,timeFact,trialFact] = viz_ktensor_freesurfer(est_factors,channelsPooled,timeEpoch,phonemeTrial,nDisp)

 figure;

   numPlots = 6;
    t = tiledlayout(nDisp,numPlots,'TileSpacing','compact');

for iCom = 1:nDisp
   
    
    elecFact(iCom,:) = est_factors.u{1}(:,iCom);
    timeFact(iCom,:) = est_factors.u{2}(:,iCom);
    trialFact(iCom,:) = est_factors.u{3}(:,iCom);
%     cfg = plot_defaults([]);
%     cfg.elec_size = 20;
%     cfg.hemisphere = 'l';
%     nexttile;
%     plot_subjs_on_average_activation(channelsPooled', elecFact(iCom,:)', 'fsaverage', cfg)
%     cfg.hemisphere = 'r';
%     nexttile;
%     plot_subjs_on_average_activation(channelsPooled', elecFact(iCom,:)', 'fsaverage', cfg)
    
   bgAx = nexttile([1 3]);

   if(iCom==1)
        title('Time Factor');
   end
   timeGamma1 = linspace(timeEpoch(1,1),timeEpoch(1,2),(timeEpoch(1,2)-timeEpoch(1,1))*200 );
   timeGamma2 = linspace(timeEpoch(2,1),timeEpoch(2,2),(timeEpoch(2,2)-timeEpoch(2,1))*200 );
   timeGamma3 = linspace(timeEpoch(3,1),timeEpoch(3,2),(timeEpoch(3,2)-timeEpoch(3,1))*200 );
%    length(timeGamma1)
%    length(timeGamma2)
%    length(timeGamma3)
%     t = tiledlayout(1,2,'TileSpacing','compact');
     %bgAx = axes(t,'XTick',[],'YTick',[],'Box','off');
    %bgAx.Layout.TileSpan = [1 2];
    bgAx.XTick = [];
    bgAx.YTick = [];
    bgAx.Box = 'off';
    ax1 =  axes(t);
    ax1.Layout.Tile = numPlots*(iCom-1)+1;
    plot(ax1,timeGamma1,timeFact(iCom,1:round(length(timeGamma1))),'LineWidth',2,'Color','k');
    xline(ax1,timeGamma1(end),':');
    ax1.Box = 'off';
    xlim(ax1,[timeGamma1(1) timeGamma1(end)])
    if(iCom==nDisp)
    xlabel(ax1, 'Auditory')
    end
 

    ax2 =  axes(t);
    ax2.Layout.Tile = numPlots*(iCom-1)+2;
    startTimePoint = round(length(timeGamma1))+1;
    plot(ax2,timeGamma2,timeFact(iCom,startTimePoint:startTimePoint+length(timeGamma2)-1),'LineWidth',2,'Color','k')
    xline(ax2,timeGamma2(1),':');
    xline(ax2,timeGamma2(end),':');
    ax2.YAxis.Visible = 'off';
    ax2.Box = 'off';
    xlim(ax2,[timeGamma2(1) timeGamma2(end)])
  
    if(iCom==nDisp)
    xlabel(ax2,'Go')
    
    end

    ax3 =  axes(t);
    ax3.Layout.Tile = numPlots*(iCom-1)+3;
    startTimePoint = startTimePoint+round(length(timeGamma2));
    plot(ax3,timeGamma3,timeFact(iCom,startTimePoint:end),'LineWidth',2,'Color','k')
    xline(ax3,timeGamma3(1),':');
    xline(ax3,timeGamma3(end),':');
    ax3.YAxis.Visible = 'off';
    ax3.Box = 'off';
    xlim(ax3,[timeGamma3(1) timeGamma3(end)])

    if(iCom==nDisp)
    xlabel(ax3,'Response Onset')
    end

% Link the axes
    linkaxes([ax1 ax2 ax3], 'y')

   

   
   nexttile;
   [p,tbl] = anova1(trialFact(iCom,:),phonemeTrial.phonemeUnit(:,1),'off');
   f1 = tbl{2,5};
   [p,tbl] = anova1(trialFact(iCom,:),phonemeTrial.phonemeUnit(:,2),'off');
   f2 = tbl{2,5};
   [p,tbl] = anova1(trialFact(iCom,:),phonemeTrial.phonemeUnit(:,3),'off');
   f3 = tbl{2,5};
   scatter(1:3, [f1 f2 f3],'filled','LineWidth',2);
   ylim([0 10])
   hold on;
   line(1:3,[f1 f2 f3],'LineWidth',2);
   
   set(gca,'xtick',1:3)
    set(gca,'XTickLabel',{'P1','P2','P3'});
    ylabel('F-statistic');
     
    if(iCom==1)
    title('Phoneme ');
    end
   
    axis square
  
   
    nexttile;
   [p,tbl] = anova1(trialFact(iCom,:),phonemeTrial.phonemeClass(:,1),'off');
   f1 = tbl{2,5};
   [p,tbl] = anova1(trialFact(iCom,:),phonemeTrial.phonemeClass(:,2),'off');
   f2 = tbl{2,5};
   [p,tbl] = anova1(trialFact(iCom,:),phonemeTrial.phonemeClass(:,3),'off');
   f3 = tbl{2,5};
   scatter(1:3, [f1 f2 f3],'filled','LineWidth',2);
   ylim([0 30])
   hold on;
   line(1:3,[f1 f2 f3],'LineWidth',2);
   
   set(gca,'xtick',1:3)
    set(gca,'XTickLabel',{'P1','P2','P3'});
    ylabel('F-statistic');
    
    if(iCom==1)
    title('Articulator ');
    end
    
    axis square
    
    
    cvcIds = find(phonemeTrial.syllableUnit(:,1)'==2);
    vcvIds = find(phonemeTrial.syllableUnit(:,1)'==1);
    for iPhon = 1:9
    cvcmdl = fitlm(trialFact(iCom,cvcIds),-log(1+phonemeTrial.phonotactic(cvcIds,iPhon)'));
    fstat_cvc(iPhon) = cvcmdl.ModelFitVsNullModel.Fstat;
    end
    cvcmdl = fitlm(trialFact(iCom,cvcIds),phonemeTrial.phonotactic(cvcIds,10)');
    fstat_cvc(10) = cvcmdl.ModelFitVsNullModel.Fstat;

    for iPhon = 1:9
    vcvmdl = fitlm(trialFact(iCom,vcvIds),-log(1+phonemeTrial.phonotactic(vcvIds,iPhon)'));
    fstat_vcv(iPhon) = vcvmdl.ModelFitVsNullModel.Fstat;
    end
    vcvmdl = fitlm(trialFact(iCom,vcvIds),phonemeTrial.phonotactic(vcvIds,10)');
    fstat_vcv(10) = vcvmdl.ModelFitVsNullModel.Fstat;

    nexttile
    bar(1:10, [fstat_cvc;fstat_vcv])
    if(iCom==nDisp)
    xlabel('Phonotactic metric')    
    end
    ylabel('F-statistic')
    axis square;
    


    


    
%    nexttile;
%    scatter((trialFact(iCom,:)),responseTime,'filled');
%    ylim([0 1.5])
%    xlabel('Trial factor');
%    ylabel('Response time (s)');
%    axis square
%    axis tight
   
   
   
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