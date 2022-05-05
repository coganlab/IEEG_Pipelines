function [chid,chmap] = stereoChanLabelExtract(labels)
k = 0;
for n = 1: length(labels)
    if(strlength(labels{n})==5)
        chid(n) = str2double(labels{n}(5));
    else
        chid(n) = str2double(labels{n}(5:6));
    end
    if(chid(n)==1)
        chmap(n)= k+1;
        k = k+1;
    else
        chmap(n)=k;
    end
end
end