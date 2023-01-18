function [fs, cs, f, var] = ftest(X, tapers, sampling, fk, pad, pval, flag)
% FTEST computes the F-statistic for sine wave in locally-white noise.
%
% [FS, CS, F, ERR] = FTEST(X, TAPERS, SAMPLING, FK, PAD, PVAL, FLAG)
%
%  Inputs:  X		=  Time series array in [Trials,Time] form.
%	    TAPERS 	=  Data tapers in [K,TIME], [N,P,K] or [N,W] form.
%			   	Defaults to [N,3,5] where N is duration of X.
%	    SAMPLING 	=  Sampling rate of time series X in Hz. 
%				Defaults to 1.
%	    FK 	 	=  Frequency range to return in Hz in
%                               either [F1,F2] or [F2] form.  
%                               In [F2] form, F1 is set to 0.
%			   	Defaults to [0,SAMPLING/2]
%	    PAD		=  Padding factor for the FFT.  
%			      	i.e. For N = 500, if PAD = 2, we pad the FFT 
%			      	to 1024 points; if PAD = 4, we pad the FFT
%			      	to 2048 points.
%				Defaults to 2.
%	   PVAL		=  P-value to calculate error bars for.
%				Defaults to 0.05 i.e. 95% confidence.
%
%	   FLAG = 0:	calculate FTEST seperately for each channel/trial.
%	   FLAG = 1:	calculate FTEST by pooling across channels/trials. 
%		Defaults to FLAG = 0;
%
%  Outputs: FS		=  F-statistic for X in [Space/Trials, Freq] form.
%  	    CS		=  Line amplitude for X in [Space/Trials,Freq] form. 
%	    F		=  Units of Frequency axis for FS and CS.
%	    ERR 	=  Error bars for CS in [Hi/Lo, Space/Trials, Freq]
%			   form given by a Jacknife-t interval for PVAL.
%

% Modification History:  
%           Written by:  Bijan Pesaran, 1997

sX = size(X);
nt = sX(2);
nch = sX(1);

%  Set the defaults

if nargin < 3 sampling = 1; end
nt = nt./sampling;
if nargin < 2 tapers = [nt,3,5]; end
if length(tapers) == 2
   n = tapers(1);
   w = tapers(2);
   p = n*w;
   k = floor(2*p-1);
   tapers = [n,p,k];
%   disp(['Using ' num2str(k) ' tapers.']);
end
if length(tapers) == 3 
   tapers(1) = tapers(1).*sampling; 
   tapers = dpsschk(tapers);
end
if nargin < 4 fk = [0,sampling./2]; end
if length(fk) == 1
    fk = [0,fk];
end
if nargin < 5 pad = 2; end
if nargin < 6 pval = 0.05;  end
if nargin < 7 flag = 0; end 

N = length(tapers(:,1));
nt = nt.*sampling;
if N ~= nt error('Error:  Length of time series and tapers must be equal'); end

K = length(tapers(1,:));

% Determine outputs
nf = max(256,pad*2.^(nextpow2(N+1)));
df = sampling./nf;
f = [0:df:sampling-df];
nfk = [min(find(f > fk(1)))-1,min(find(f > fk(2))-1)];
f = f(nfk(1):nfk(2));
dof = 2.*nch.*K;

errorchk = 0;
if nargout > 3 errorchk = 1; end

Hk0 = sum(tapers(:,1:2:K));

%Hk = fft(tapers(:,:),nf)';
%Hk = Hk(:,nfk(1):nfk(2));

if ~flag		% No pooling across trials
  fs = zeros(nch, diff(nfk)+1);
  cs = zeros(nch, diff(nfk)+1);
  var = zeros(nch, diff(nfk)+1);
%  err = zeros(2, nch, diff(nfk)+1);
  for ch = 1:nch
     tmp = (X(ch,:) - mean(X(ch,:)))';
     xk = fft(tapers(:,1:K).*tmp(:,ones(1,K)),nf)';
     xk = xk(:,nfk(1):nfk(2));
     Sk = xk.*conj(xk);
     sp = sum(Sk);
     cs(ch,:) = (xk(1:2:K,:)'*Hk0'/sum(Hk0.^2))';
     num = (cs(ch,:).*conj(cs(ch,:)));
     fs(ch,:) = (K-1)*num./(sp/sum(Hk0.^2)-num);
     var(ch,:) = (sp./sum(Hk0.^2)-num)./K;
%     yk = xk - cs(ones(1,K),f1).*Hk(:,1).*...
%	exp(-2.*pi.*f1./nf);      
%    spec(ch,:) = mean(Sk,1);
%     if errorchk	%  Estimate error bars
%       for ik = 1:K
%         indices = setdiff([1:K],[ik]);
%         xj = xk(indices,:);
%         jlsp(ik,:) = log(mean(abs(xj).^2,1));
%       end
%       lsig = sqrt(K-1).*std(jlsp,1);
%       crit = tinv(1-pval./2,dof-1);   %   Determine the scaling
%        err(1,ch,:) = exp(log(spec(ch,:))+crit.*lsig);
%       err(2,ch,:) = exp(log(spec(ch,:))-crit.*lsig);
%     end
   end
end
