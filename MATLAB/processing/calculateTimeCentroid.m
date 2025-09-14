function [time_centroid] = calculateTimeCentroid(time,values)
%CALCULATETIMECENTROID Summary of this function goes here
%   Detailed explanation goes here
% Calculate the total area under the curve (AUC) using the trapezoidal rule
total_area = trapz(time, values);

% Find the time centroid
cumulative_area = cumtrapz(time, values);
centroid_index = find(cumulative_area >= total_area/2, 1, 'first');
time_centroid = time(centroid_index);

%fprintf('The time centroid is at t = %f\n', time_centroid);
end

