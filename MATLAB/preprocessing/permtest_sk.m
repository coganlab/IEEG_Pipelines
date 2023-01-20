function p = permtest_sk(sample1,sample2,numperm)
    samples = [sample1 sample2];
    samplediff = (median(sample1)-median(sample2));
    sampdiffshuff = zeros(1,numperm);
    for n = 1:numperm
        sampshuff = samples(randperm(length(samples)));
        sampdiffshuff(n) = (median(sampshuff(1:length(sampshuff)/2))-median(sampshuff(length(sampshuff)/2+1:end)));
    end    
    p = length(find((sampdiffshuff)>(samplediff)))/numperm;
end