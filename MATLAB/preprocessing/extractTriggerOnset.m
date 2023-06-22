% extractTriggerOnset - Extracts trigger onset locations from a signal.
%
% Syntax: locs = extractTriggerOnset(trigger, fs)
%
% Inputs:
%   trigger - Signal containing the trigger information
%   fs      - Sampling frequency of the trigger signal (in Hz)
%
% Output:
%   locs    - Array of trigger onset locations (in seconds)
%



function locs = extractTriggerOnset(trigger, fs)
    time = [0:length(trigger)-1]./fs;
    
    % Plot the trigger signal
    figure;
    plot(time, trigger);
    
    % Prompt user for input
    tw(1) = input('Enter the starting time: ');
    tw(2) = input('Enter the ending time: ');
    amp = input('Enter the amplitude: ');
    mpd = input('Enter the minimum peak distance: ');
    
    % Select the time range for analysis
    timeSelectInd = time >= tw(1) & time <= tw(2);
    timeSelect = time(timeSelectInd);
    
    % Find peaks in the trigger signal within the selected time range
    [~, locs] = findpeaks(trigger(timeSelectInd), fs, 'MinPeakDistance', mpd, 'MinPeakHeight', amp);
    
    % Adjust the peak locations to the absolute time scale
    locs = locs + tw(1);
    
    % Plot the selected time range with peak locations
    figure;
    plot(timeSelect, trigger(timeSelectInd));
    hold on;
    scatter(locs, max(trigger(timeSelectInd)) .* ones(1, length(locs)));
end
