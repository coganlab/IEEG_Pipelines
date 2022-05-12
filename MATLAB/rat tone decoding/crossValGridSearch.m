function loss = crossValGridSearch(signal,labels,varargin)
        
    cvp = cvpartition(labels,'KFold',10,'Stratify',true);
    linearTemplate = templateDiscriminant('DiscrimType','pseudolinear');
    score = []; scoreF = []; explained = [];
    parfor i = 1:size(signal,1)
        sig2analyze = squeeze(signal(i,:,:));
        [~,score(i,:,:),~,~,explained(i,:)] = pca(sig2analyze,'Centered',false);
    end
    if(nargin==3)
        for v = 1:numDim        
            %scoreF = (mean(score(:,:,1:v),3))';   
            v
            for i =  1:size(signal,1)
             for tr = 1:size(score,2)
                scoreF(tr,i) = norm(squeeze(score(i,tr,1:v)));
             end
            end
            linearModel = fitcecoc((scoreF),labels,'Coding','onevsall','Learners',linearTemplate,'CrossVal','on','CVPartition',cvp,'Options',statset('UseParallel',true));
            loss(v) = kfoldLoss(linearModel);
        end
    end
    if(nargin==4)
        varVect = [10:2:98];
     for v = 1:length(varVect)
         v
        for i =  1:size(signal,1)
            id = find(cumsum(explained(i,:))>=varVect(v),1);
            %scoreF(:,i) = (sum(score(i,:,1:id),3));
            for tr = 1:size(score,2)
                scoreF(tr,i) = norm(squeeze(score(i,tr,1:id)));
            end
            
        end
        linearModel = fitcecoc(zscore(scoreF),labels,'Coding','onevsall','Learners',linearTemplate,'CrossVal','on','CVPartition',cvp,'Options',statset('UseParallel',true));
        loss(v) = kfoldLoss(linearModel);
     end
    end
end