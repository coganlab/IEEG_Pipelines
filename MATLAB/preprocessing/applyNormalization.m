function ieegStructNorm = applyNormalization(ieegStruct,normFactor,normType)
%APPLYNORMALIZATION Summary of this function goes here
%   Detailed explanation goes here
ieegStructNorm = ieegStruct;
for iChan = 1:size(normFactor,1)
    if(normType==1) % z-score
        ieegStructNorm.data(iChan,:,:) = (ieegStructNorm.data(iChan,:,:)-normFactor(iChan,1))./normFactor(iChan,2); 
        end
        if(normType==2) % mean-normalization
            ieegStructNorm.data(iChan,:,:) = (ieegStructNorm.data(iChan,:,:)-normFactor(iChan,1));
        end
        if(normType==3) % percentage absolute relative baseline
            ieegStructNorm.data(iChan,:,:) = (ieegStructNorm.data(iChan,:,:)-normFactor(iChan,1))./normFactor(iChan,1);
        end
        if(normType==4) % percentage relative baseline
            ieegStructNorm.data(iChan,:,:) = (ieegStructNorm.data(iChan,:,:)./normFactor(iChan,1));
        end
        if(normType==5) % Log-transform baseline
            ieegStructNorm.data(iChan,:,:) = 10.*log10(ieegStructNorm.data(iChan,:,:)./normFactor(iChan,1));
        end
        if(normType==6) % Normed baseline
            ieegStructNorm.data(iChan,:,:) = (ieegStructNorm.data(iChan,:,:)-normFactor(iChan,1))./(ieegStructNorm.data(iChan,:,:)+normFactor(iChan,1));
        end
end
end

