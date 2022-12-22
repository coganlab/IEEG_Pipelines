function viz_nnmf_factors(chanMap,selectedChannels,timeGammaPerc,timeGammaProd,factorChan,factorTime,clusterScore1, clusterScore2)


figure;

   
    t = tiledlayout(1,5,'TileSpacing','compact');



    nexttile;
    chanView(factorChan,chanMap,selectedChannels,isnan(chanMap),[],[],[],[]);
    title('Spatial Clusters');
    set(gca,'xtick',[])
    set(gca,'ytick',[])
    colormap(lines(length(unique(factorChan))));
    axis equal
    axis tight
    
    
    
    bgAx = nexttile([1 2]);
   set(gca,'FontSize',12);
   
%     t = tiledlayout(1,2,'TileSpacing','compact');
     %bgAx = axes(t,'XTick',[],'YTick',[],'Box','off');
    %bgAx.Layout.TileSpan = [1 2];
    bgAx.XTick = [];
    bgAx.YTick = [];
    bgAx.Box = 'off';
    ax1 =  axes(t);
    ax1.Layout.Tile = 2;
    plot(ax1,timeGammaPerc,factorTime(:,1:round(length(factorTime)/2)),'LineWidth',2);
    xline(ax1,timeGammaPerc(end),':');
    ax1.Box = 'off';
    xlim(ax1,[timeGammaPerc(1) timeGammaPerc(end)])
    
    xlabel(ax1, 'Auditory')
    
    set(gca,'FontSize',12);

    ax2 =  axes(t);
    ax2.Layout.Tile = 3;
    plot(ax2,timeGammaProd,factorTime(:,round(length(factorTime)/2)+1:end),'LineWidth',2)
    xline(ax2,timeGammaProd(1),':');
    ax2.YAxis.Visible = 'off';
    ax2.Box = 'off';
    xlim(ax2,[timeGammaProd(1) timeGammaProd(end)])
    
    xlabel(ax2,'Production')
    set(gca,'FontSize',12);
    
    
    nexttile; h = boxplot(clusterScore1,factorChan,'symbol','');
     set(h,{'linew'},{2})
     xlabel('Clusters');
     set(gca,'FontSize',15);
     
      nexttile; h = boxplot(clusterScore2,factorChan,'symbol','');
     set(h,{'linew'},{2})
     xlabel('Clusters');
     set(gca,'FontSize',15);
    
    
end