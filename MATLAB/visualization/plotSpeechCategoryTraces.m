function plotSpeechCategoryTraces(speechCatData, tw, catLabels, catIDs, chanMap, quadZoom)
    switch nargin
    case 1
        tw = [-2, 2]
        catLabels = {'a', 'ae', 'i', 'u', 'b', 'p', 'v', 'g', 'k'}; % default to phonemes
        catIDs = 1;
        chanMap = reshape(1:128, 16, 8); % no specified channel map
        quadZoom = 0;
    case 2
        catLabels = {'a', 'ae', 'i', 'u', 'b', 'p', 'v', 'g', 'k'}; % default to phonemes
        catIDs = 1;
        chanMap = reshape(1:128, 16, 8); % no specified channel map
        quadZoom = 0;
    case 3
        catIDs = 1;
        chanMap = reshape(1:128, 16, 8); % no specified channel map
        quadZoom = 0;
    case 4
        chanMap = reshape(1:128, 16, 8); % no specified channel map
        quadZoom = 0;
    case 5
        quadZoom = 0;
end

tPlot = linspace(tw(1), tw(2), size(speechCatData{1}, 3));


catStr = '';
legendLabels = cell(1, length(catIDs));
for iCat=1:length(catIDs)
    catStr = strcat(catStr, "'"+catLabels{catIDs(iCat)}+"', ");
    legendLabels{iCat} = catLabels{catIDs(iCat)};
end
titleStr = convertStringsToChars(strcat("Voltage Traces by Channel for Classes(s): ", catStr));
plotLabels = {'Time Relative to Speech Onset (s)', 'Normalized Amplitude', ...
                titleStr(1:end-2), legendLabels};
% data = squeeze(mean(speechCatData{catIDs}, 2)); % average across trials
dataCell = cell(1, length(catIDs));
for iCat = 1:length(catIDs)
    dataCell{iCat} = speechCatData{catIDs(iCat)};
end
plotChanMap128(dataCell, chanMap, tPlot, plotLabels, 1, quadZoom)
