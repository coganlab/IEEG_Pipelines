function matrixPoints = matrixSubSample(chanMap,window,isOverlap)
[m, n] = size(chanMap);
mWindow = window(1);
nWindow = window(2);
matrixPoints = [];
numWin = 1;
if(isOverlap)
    for iRow = 1:m-mWindow+1
        for iCol = 1:n-nWindow+1        
            matrixPointsMap = chanMap(iRow:iRow+mWindow-1,iCol:iCol+nWindow-1);
            matrixPoints(numWin,:) = matrixPointsMap(:)';
            numWin = numWin+1;
        end
    end
else
    for iRow = 1:mWindow:m-mWindow+1
        for iCol = 1:nWindow:n-nWindow+1        
            matrixPointsMap = chanMap(iRow:iRow+mWindow-1,iCol:iCol+nWindow-1);
            matrixPoints(numWin,:) = matrixPointsMap(:)';
            numWin = numWin+1;
        end
    end
end