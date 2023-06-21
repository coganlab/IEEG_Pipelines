% Function: matrixSubSample
% Description: Performs matrix subsampling based on a window size and overlap.
% Inputs:
%   - chanMap: Input matrix for subsampling
%   - window: Window size for subsampling (1x2 array)
%   - isOverlap: Flag indicating whether to use overlap or not
% Outputs:
%   - matrixPoints: Subsampled matrix points

function matrixPoints = matrixSubSample(chanMap, window, isOverlap)
    [m, n] = size(chanMap); % Get the size of the input matrix
    mWindow = window(1); % Height of the window
    nWindow = window(2); % Width of the window
    matrixPoints = []; % Initialize the subsampled matrix points
    numWin = 1; % Counter for the number of windows
    
    if (isOverlap) % If overlap is enabled
        for iRow = 1:m - mWindow + 1 % Loop through rows with overlap
            for iCol = 1:n - nWindow + 1 % Loop through columns with overlap
                matrixPointsMap = chanMap(iRow:iRow + mWindow - 1, iCol:iCol + nWindow - 1);
                matrixPoints(numWin, :) = matrixPointsMap(:)'; % Reshape the window to a row vector and store in matrixPoints
                numWin = numWin + 1; % Increment the window counter
            end
        end
    else % If overlap is disabled
        for iRow = 1:mWindow:m - mWindow + 1 % Loop through rows without overlap
            for iCol = 1:nWindow:n - nWindow + 1 % Loop through columns without overlap
                matrixPointsMap = chanMap(iRow:iRow + mWindow - 1, iCol:iCol + nWindow - 1);
                matrixPoints(numWin, :) = matrixPointsMap(:)'; % Reshape the window to a row vector and store in matrixPoints
                numWin = numWin + 1; % Increment the window counter
            end
        end
    end
end
