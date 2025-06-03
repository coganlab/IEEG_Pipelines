function matrixPoints = matrixSubSample(chanMap,window,isOverlap,isCenter)
[m, n] = size(chanMap);
mWindow = window(1);
nWindow = window(2);
matrixPoints = [];
numWin = 1;

% center window in matrix if desired
if isCenter
    mStart = idivide(mod(m, mWindow), int16(2));
    nStart = idivide(mod(n, nWindow), int16(2));
else
    mStart = 0;
    nStart = 0;
end

if(isOverlap)
    for iRow = 1:m-mWindow+1
        for iCol = 1:n-nWindow+1        
            matrixPointsMap = chanMap((mStart+iRow):(mStart+iRow+mWindow-1),(nStart+iCol):(nStart+iCol+nWindow-1));
            matrixPoints(numWin,:) = matrixPointsMap(:)';
            numWin = numWin+1;
        end
    end
else
    for iRow = 1:mWindow:m-mWindow+1
        for iCol = 1:nWindow:n-nWindow+1        
            matrixPointsMap = chanMap((mStart+iRow):(mStart+iRow+mWindow-1),(nStart+iCol):(nStart+iCol+nWindow-1));
            matrixPoints(numWin,:) = matrixPointsMap(:)';
            numWin = numWin+1;
        end
    end
end