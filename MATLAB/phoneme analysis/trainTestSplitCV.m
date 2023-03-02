function cvp = trainTestSplitCV(labels,numFolds)
       
    if(numFolds>0)
        cvp = cvpartition(labels,'KFold',numFolds,'Stratify',true);
    else
        cvp = cvpartition(labels,'LeaveOut');
    end

end