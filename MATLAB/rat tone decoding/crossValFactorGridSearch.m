function loss = crossValFactorGridSearch(signal,labels,varargin)
    cvp = cvpartition(labels,'KFold',6,'Stratify',true);
    linearTemplate = templateDiscriminant('DiscrimType','pseudolinear');
    for f = 1:10
        score = []; scoreF = []; explained = []; F = [];
        parfor i = 1:size(signal,1)
            sig2analyze = squeeze(signal(i,:,:));
            covMat = cov1para(sig2analyze);
            [lambda,~,~,stats,F(:,:,i)] = factoranupdate((sig2analyze),f,'scores','wls','rotate','varimax','maxit',1000);
            figure;
            subplot(2,1,1);
            imagesc(sig2analyze);
            ylabel('Trials');
            subplot(2,1,2);
            plot(lambda);
            xlabel('Time (samples)');
            legend('Factor 1','Factor 2','Factor 3','Factor 4')
    %         for tr = 1:size(signal,2)
    %             scoreF(tr,i) = (F(tr,2,i));
    %         end
        end
        %scoreF = squeeze(mean(F,2));
        %scoreF = squeeze(F(:,1,:));
       scoreF = reshape(F,size(F,1),size(F,2)*size(F,3));
        linearModel = fitcecoc((scoreF),labels,'Coding','onevsall','Learners',linearTemplate,'CrossVal','on','CVPartition',cvp,'Options',statset('UseParallel',true));
                loss(f) = kfoldLoss(linearModel);
                YHat = kfoldPredict(linearModel);
        tone = unique(labels);
        C = confusionmat(labels,YHat);
        figure;
        confusionchart(C,tone);
        title(num2str(length(find(YHat==labels))*100/length(labels)));
    end
end