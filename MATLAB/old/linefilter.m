function [Y, fs, cs, f0, X_fit] = linefilter(X, tapers, sampling, fks, pad);
%  LINEFILTER filters a harmonic line from input univariate time
%  series.  It detects the single strongest line in each of a number of
%  frequency bands.
%
%  [Y, FS, CS, F0] = LINEFILTER(X, TAPERS, SAMPLING, FKS)
%
%  Inputs:  X = Univariate time series
%           TAPERS = 
%           SAMPLING
%           FKS  = Frequency ranges.
%               For example:[[40,90];[100,150];[150,200];[200,400]] filters
%                   between 40 and 90 Hz, 100 and 150 Hz, 150 and 200 Hz
%                   and so on.
%
%  Outputs: Y  = Univariate time series
%           FS = F-spectrum for harmonic analysis
%           CS = Complex amplitude spectrum for harmonic analysis
%           F0 = Line frequency removed
%

%  Written by:  Bijan Pesaran
%

nt = size(X,2);


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
if nargin < 4 fks = [0,sampling./2]; end
if length(fks) == 1
    fks = [0,fks];
end
if nargin < 5 pad = 8; end
N = length(tapers(:,1));
nt = nt.*sampling;
if N ~= nt error('Error:  Length of time series and tapers must be equal'); end

K = length(tapers(1,:));

[fs, cs, f] = ftest(X, tapers, sampling, max(max(fks)), pad);

for iFk = 1:size(fks,1)
    f_ind = find(f>fks(iFk,1) & f<fks(iFk,2));
    [f_val,ind] = max(fs(f_ind));
    f0 = f(f_ind(ind));
    pdf = 1-fcdf(f_val,2,2*K-2);

    if pdf < 1./N
      %   disp(['Significant line (' num2str(1-pdf) ') found at ' ...
      % 	num2str(f0) ' and removed.']);
        X_fit(iFk,:) = 2.*real(cs(f_ind(ind)).*exp(-2.*pi.*complex(0,1).*f0.*[0:N-1]./sampling));
    else
        X_fit(iFk,:) = zeros(1,length(X));
    end
end

X_fit = sum(X_fit,1);
Y = X - X_fit;