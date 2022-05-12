function p = permtest_sk(sample1,sample2,numperm)
    samples = [sample1 sample2];
    samplediff = (mean(sample1)-mean(sample2));
    sampdiffshuff = zeros(1,numperm);
    for n = 1:numperm
        sampshuff = samples(randperm(length(samples)));
        sampdiffshuff(n) = (mean(sampshuff(1:length(sampshuff)/2))-mean(sampshuff(length(sampshuff)/2+1:end)));
    end    
    p = length(find((sampdiffshuff)>(samplediff)))/numperm;
end