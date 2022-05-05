function [S,w] = lrrSegment(ieeg)
% The function performs the linear regression referencing on ECoG data with
% multiple channels. 
% Reference
% Young, D., et al. "Signal processing methods for reducing artifacts in 
% microelectrode brain recordings caused by functional electrical stimulation."
% Journal of neural engineering 15.2 (2018): 026014.
% Created by Kumar Duraivel for Viventi and Cogan lab
% Input
% x(n x trials x time) - Input signal with 'n' channels and 't' timepoints
% Output
% S(n x t) - Linear regression referenced signal 
% w(n x trial x n-1) - Least square weights

n = size(ieeg,1); % number of channels
c = 1:n;
for tr = 1:size(ieeg,2)
    tr
    ieegtrial = squeeze(ieeg(:,tr,:));
    for i = 1:n
        [cw(i,:),lw(i,:)] = wavedec(ieegtrial(i,:),10,'db2');
    end
    Strial = zeros(n,size(ieeg,3));
    for wd = 1:11
        parfor i = 1:n
            if(wd~=11)
                x(i,:) = wrcoef('d',cw(i,:),lw(i,:),'db2',wd);
            else
                x(i,:) = wrcoef('a',cw(i,:),lw(i,:),'db2',wd-1);
            end
        end
        parfor i = 1:n % Iterating through channels        
            R = (x(i,:))'; % Reference channel
            X = ((x(setdiff(c,i),:)))'; % Channels to regress
            wtrial = ((X'*X)\(X'*R))'; % Least square weights estimation
            %Strialwd(i,:) = R' - wtrial*X'; % LRR referencing
            Strialwd(i,:) = wtrial*X'; % LRR referencing
            wtrialu(wd,i,:) = wtrial;
        end
        Strial = Strial+Strialwd;
    end
    S(:,tr,:) = Strial;
    w(:,:,tr,:) = wtrialu;
end
end
%%
% ieegcheck = squeeze(ieeg(1,1,:));
% %[w,l] = wavedec(ieegcheck,10,'db2');
% [w,l] = wavedec(ieegcheck',10,'db2');
% approx = appcoef(w,l,'db2');
% figure;
% d = [];
% for i = 1:6
% d(i,:) = wrcoef('d',w,l,'db2',i);
% [psd,f] = pwelch(d(i,:),[],[],[],fs);
% semilogx(f,log(psd));
% hold on;
% end
% a = wrcoef('a',w,l,'db2',6);
% sigRecon = sum(d,1)+a;
% [psdclean,fclean] = pwelch(ieegcheck,[],[],[],fs);
% [psdrecon,fclean] = pwelch(sigRecon,[],[],[],fs);
% figure;
% loglog(fclean,psdclean);
% hold on;
% loglog(fclean,psdrecon);
% 
% subplot(2,1,1);
% plot(ieegcheck);
% subplot(2,1,2);
% plot(a3);