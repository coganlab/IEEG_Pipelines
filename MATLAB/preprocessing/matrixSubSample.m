% matrixSubSample - Subsample a matrix by dividing it into smaller windowed segments.
%
% Syntax: matrixPoints = matrixSubSample(chanMap, window, isOverlap)
%
% Inputs:
%   chanMap     - Input matrix (m x n) to be subsampled
%   window      - Window size [mWindow, nWindow] for the subsampling (in number of elements)
%   isOverlap   - Flag indicating whether to use overlapping windows (1) or non-overlapping windows (0)
%
% Output:
%   matrixPoints- Subsampled matrix points where each row represents a windowed segment
%
% Example:
%   chanMatrix = randn(100, 100); % Input matrix of size 100x100
%   windowSize = [10, 10]; % Window size of 10x10
%   overlap = 1; % Use overlapping windows
%   subsampledMatrix = matrixSubSample(chanMatrix, windowSize, overlap);
%

function matrixPoints = matrixSubSample(chanMap, window, isOverlap)
    [m, n] = size(chanMap);
    mWindow = window(1);
    nWindow = window(2);
    matrixPoints = [];
    numWin = 1;
    
    if isOverlap
        % Use overlapping windows
        for iRow = 1:m - mWindow + 1
            for iCol = 1:n - nWindow + 1
                matrixPointsMap = chanMap(iRow:iRow + mWindow - 1, iCol:iCol + nWindow - 1);
                matrixPoints(numWin, :) = matrixPointsMap(:)';
                numWin = numWin + 1;
            end
        end
    else
        % Use non-overlapping windows
        for iRow = 1:mWindow:m - mWindow + 1
            for iCol = 1:nWindow:n - nWindow + 1
                matrixPointsMap = chanMap(iRow:iRow + mWindow - 1, iCol:iCol + nWindow - 1);
                matrixPoints(numWin, :) = matrixPointsMap(:)';
                numWin = numWin + 1;
            end
        end
    end
end
