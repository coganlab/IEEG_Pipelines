function specChanMapNoNorm(spec, chanMap, selectedChannels, tw, fw, etw, efw, cval, isIndView)
% specChanMapNoNorm - Plot spectrograms without normalization
%
% Inputs:
%   spec - Cell array of spectrograms
%   chanMap - Channel mapping
%   selectedChannels - Selected channels
%   tw - Time window [start, end]
%   fw - Frequency window [start, end]
%   etw - Time window for extraction [start, end]
%   efw - Frequency window for extraction [start, end]
%   cval - Color axis limits [min, max]
%   isIndView - Flag indicating whether to plot individual view
%
% Note: This function plots spectrograms without normalization.
%

tspec = linspace(tw(1), tw(2), size(spec{1}, 2)); % Time axis
Fspec = linspace(fw(1), fw(2), size(spec{1}, 1)); % Frequency axis
etspec = find(tspec >= etw(1) & tspec <= etw(2)); % Indices of time window for extraction
efspec = find(Fspec >= efw(1) & Fspec <= efw(2)); % Indices of frequency window for extraction

if isIndView
    for i = isIndView
        specMean = spec{i}(efspec, etspec); % Extract spectrogram within the specified windows
        imagesc(tspec(etspec), Fspec(efspec), specMean); % Plot spectrogram
        caxis(cval);
        colormap(jet(4096));
        set(gca, 'YDir', 'normal');
        xlabel('Time (s)');
        ylabel('Frequency (Hz)');
        title(strcat('Channel: ', num2str(selectedChannels(i))));
    end
else
    figure;
    for i = 1:length(spec)
        specMean = spec{i}(efspec, etspec); % Extract spectrogram within the specified windows
        subplot(size(chanMap, 1), size(chanMap, 2), find(ismember(chanMap', selectedChannels(i))));
        imagesc(tspec(etspec), Fspec(efspec), specMean); % Plot spectrogram
        caxis(cval);
        colormap(jet(4096));
        set(gca, 'YDir', 'normal');
        axis off;
        set(gca, 'xtick', [], 'ytick', [])
        colormap(jet(4096));
    end
end
