function plotChanMap128(dataCell, chanMap, xData, plotLabels, errorPlot, quadZoom)
    switch nargin
    case 1
        chanMap = reshape(1:128, 16, 8); % no specified channel map
        xData = 1:size(data, 3); % number of time points
        plotLabels = {"", "", "", ""}; % no labels
        errorPlot = 1; % default to plotting error
        quadZoom = 0; % default to no quadrant view
    case 2
        xData = 1:size(data, 3); % number of time points
        plotLabels = {"", "", "", ""}; % no labels
        errorPlot = 1; % default to plotting error
        quadZoom = 0; % default to no quadrant view
    case 3
        plotLabels = {"", "", "", ""}; % no labels
        errorPlot = 1; % default to plotting error
        quadZoom = 0; % default to no quadrant view
    case 4
        errorPlot = 1; % default to plotting error
        quadZoom = 0; % default to no quadrant view
    case 5
        quadZoom = 0; % default to no quadrant view
    end

    numChan = 128;
    numToPlot = length(dataCell);
    chanRows = size(chanMap, 1);
    chanCols = size(chanMap, 2);

    % check number of timepoints in data matches length of xData
    for iCell = 1:numToPlot
        assert(size(dataCell{iCell}, 3) == length(xData) && numel(xData) == length(xData), 'Number of timepoints in data does not match length of xData.')
        % check number of channels in data is correct
        assert(size(dataCell{iCell}, 1) == numChan, 'Number of channels in data is not 128.')
    end

    meanByTrialCell = cell(1, numToPlot);
    errorCell = cell(1, numToPlot);
    yLower = Inf;
    yUpper = -Inf;
    for iCell = 1:numToPlot
        meanByTrialCell{iCell} = squeeze(mean(dataCell{iCell}, 2));
        errorCell{iCell} = squeeze(1.96 * std(dataCell{iCell}, 0, 2) / sqrt(size(dataCell{iCell}, 2) - 1));
        if max(meanByTrialCell{iCell} + errorCell{iCell}) > yUpper
            yUpper = max(meanByTrialCell{iCell} + errorCell{iCell}, [], 'all');
        end
        if min(meanByTrialCell{iCell} - errorCell{iCell}) < yLower
            yLower = min(meanByTrialCell{iCell} - errorCell{iCell}, [], 'all');
        end
    end
    
    xDataError = [xData, fliplr(xData)];
    fillAlpha = 0.25;
    cc = distinguishable_colors(numToPlot); % from MATLAB File Exchange ("Generate maximally perceptually-distinct colors")

    pGrid = numSubplots(numChan);
    f = figure;
    set(f, 'Position', [100, 100, 1300, 750])
    for iChan = 1:numChan
        subplot(pGrid(1), pGrid(2), find(chanMap == iChan))
        hold on
        for iCell = 1:numToPlot
            dataMeanByTrial = meanByTrialCell{iCell};
            plot(xData, dataMeanByTrial(iChan, :), 'color', cc(iCell, :))
            if errorPlot
                error = errorCell{iCell};
                patch = fill(xDataError, [dataMeanByTrial(iChan, :) + error(iChan, :), fliplr(dataMeanByTrial(iChan, :) - error(iChan, :))], cc(iCell, :));
                set(patch, 'edgecolor', 'none');
                set(patch, 'FaceAlpha', fillAlpha);
            end
        end
        hold off
        if all([~isinf(yLower) ~isinf(yUpper)])
            ylim([yLower, yUpper])
            % ylim([-35, 35]) % looking at noise in raw data
        end
        drawnow
    end

    % format legend for multiple data series
    if numToPlot > 1
        legendLabels = plotLabels{4};
        subplot(pGrid(1), pGrid(2), numChan)
        lgnd = legend('');
        hold on
        for iLeg = 1:numToPlot
            plot([NaN NaN], [NaN NaN], 'color', cc(iLeg, :), 'DisplayName', legendLabels{iLeg})
        end
        hold off
        lgnd.Position(1) = 0.92;
        lgnd.Position(2) = 0.4;
    end

    annotation('line', [0.515 0.515], [0.09 0.93], 'color', 'k', 'linewidth', 1.25)
    annotation('line', [0.12 0.91], [0.51 0.51], 'color', 'k', 'linewidth', 1.25)

    han=axes(f,'visible','off'); 
    han.Title.Visible='on';
    han.XLabel.Visible='on';
    han.YLabel.Visible='on';
    xlabel(han, plotLabels{1});
    ylabel(han, plotLabels{2});
    title(han, plotLabels{3});

    % create extra figures zoomed-in on quadrants of grid to make visualization easier
    if quadZoom
        for iQuad = 1:4
            [quadCol, quadRow] = ind2sub([2, 2], iQuad);
            chanMapTranspose = chanMap';
            currQuad = chanMapTranspose((chanCols/2)*(quadRow-1)+1:(chanCols/2)*quadRow,...
                                        (chanRows/2)*(quadCol-1)+1:(chanRows/2)*quadCol);

            qGrid = numSubplots(numel(currQuad));
            qf = figure;
            set(qf, 'Position', [100, 100, 1300, 750])
            for iChan = 1:numChan
                if ~ismember(iChan, currQuad)
                    continue
                end
                subplot(qGrid(1), qGrid(2), find(currQuad' == iChan))
                hold on
                for iCell = 1:numToPlot
                    dataMeanByTrial = meanByTrialCell{iCell};
                    plot(xData, dataMeanByTrial(iChan, :), 'color', cc(iCell, :))
                    if errorPlot
                        error = errorCell{iCell};
                        patch = fill(xDataError,...
                                    [dataMeanByTrial(iChan, :) + error(iChan, :),...
                                     fliplr(dataMeanByTrial(iChan, :) - error(iChan, :))],...
                                    cc(iCell, :));
                        set(patch, 'edgecolor', 'none');
                        set(patch, 'FaceAlpha', fillAlpha);
                    end
                end
                hold off
                if all([~isinf(yLower) ~isinf(yUpper)])
                    ylim([yLower, yUpper])
                    % ylim([-35, 35]) % looking at noise in raw data
                end
                drawnow
            end

            % format legend for multiple data series
            if numToPlot > 1
                legendLabels = plotLabels{4};
                subplot(qGrid(1), qGrid(2), numel(currQuad))
                lgnd = legend('');
                hold on
                for iLeg = 1:numToPlot
                    plot([NaN NaN], [NaN NaN], 'color', cc(iLeg, :), 'DisplayName', legendLabels{iLeg})
                end
                hold off
                lgnd.Position(1) = 0.92;
                lgnd.Position(2) = 0.4;
            end

            han=axes(qf,'visible','off'); 
            han.Title.Visible='on';
            han.XLabel.Visible='on';
            han.YLabel.Visible='on';
            xlabel(han, plotLabels{1});
            ylabel(han, plotLabels{2});

            chanStr = '';
            if [quadRow, quadCol] == [1, 1]
                chanStr = 'Top Left';
            elseif [quadRow, quadCol] == [1, 2]
                chanStr = 'Top Right';
            elseif [quadRow, quadCol] == [2, 1]
                chanStr = 'Bottom Left';
            elseif [quadRow, quadCol] == [2, 2]
                chanStr = 'Bottom Right';
            end
            title(han, strcat(chanStr, ": ", plotLabels{3}));

        end
    end

    