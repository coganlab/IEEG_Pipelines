function [indLabAcc,indLabChannel] = extractElectrodeImportance(Call,CChannel,minLossIdAll,minLossIdChannel)
    Coverall = Call{minLossIdAll};
    indLabAcc = diag(Coverall)./sum(Coverall,2);
    for i = 1:size(CChannel,2)
        CoverallChannel = CChannel{i}{minLossIdChannel(i)};
        indLabChannel(i,:) = diag(CoverallChannel)./sum(CoverallChannel,2);
    end
    
end