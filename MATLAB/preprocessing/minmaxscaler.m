function minmaxx = minmaxscaler(x)
% minmaxscaler - Perform min-max scaling on the input data.
%
% Syntax: minmaxx = minmaxscaler(x)
%
% Inputs:
%   x           - Input data to be scaled (1 x n) array
%
% Outputs:
%   minmaxx     - Min-max scaled data (1 x n) array
%
% Example:
%   data = [10, 20, 30, 40, 50]; % Example input data
%   scaledData = minmaxscaler(data);
%


minmaxx = (x - min(x)) / (max(x) - min(x)); % Perform min-max scaling on the input data

end
