function [lossMod,Cmat,yhat,aucVect] = lindscrDecode(X_train, y_train, X_test, y_test, isauc)

    linearModel = fitcdiscr(X_train,y_train,'CrossVal','off','DiscrimType','linear'); 

    [yhat,yscore] = predict(linearModel, X_test);
    labUnique = unique(y_test);
    aucVect = zeros(1,length(labUnique));
    if(isauc)
        for t = 1:length(labUnique)
            [~,~,~,aucVect(t)] = perfcurve(y_test,yscore(:,t),labUnique(t));
        end
    end
    lossMod = loss(linearModel,X_test,y_test);
    Cmat =  confusionmat(y_test,yhat);

end