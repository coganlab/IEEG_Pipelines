function [dh,dh2] = EcogExtractHighGammaTrial (d, infs,outfs, freqRange,tw,etw,normFactor,normType)
% d: recorded data, channel x sample
% infs: sampling rate of the data
% outfs: sampling rate of output
% freqRange: bandpass filtering range, e.g. [70 150]
% tw: time-window of the data e.g [-1 4]
% etw: output time-window (to remove filter artefacts) e.g. [-0.5 3.5]
% normFactor: Normalizing factor to calculate z-score  e.g. [mu sigma]


% dh: channel x sample x center-frequencies
% dh2: channel x sample (recommended)

% Nima, nimail@gmail.com
% Neural Acoustic Processing Lab, 
% Columbia University, naplab.ee.columbia.edu
% Updated by Kumar, sd355@duke.edu
% Cogan lab

%defaultfs = 400; % Hz
%freqRange=[70 150];
fs = infs;
time = linspace(tw(1),tw(2),size(d,2));
eTime = time>=etw(1)&time<=etw(2);
% if infs ~= defaultfs
%     d = resample(d',defaultfs,infs)';
%     fs = defaultfs;
% end

% apply notch filter:
% notchFreq=60;
% while notchFreq<fs/2
%     [b,a]=fir2(1000,[0 notchFreq-1 notchFreq-.5 notchFreq+.5 notchFreq+1 fs/2]/(fs/2),[1 1 0 0 1 1 ]);
%     d=filtfilt(b,a,d')';
%     notchFreq=notchFreq+60;
% end

% calculate hilbert envelope:
[dh,cfs,sigma_fs] = CUprocessingHilbertTransform_filterbankGUI(d, fs, freqRange);
% dh3 = mean(log10(dh.^2),3);
% dh3 = mapstd(dh3(:,eTime));
%
dh2 = mean(abs(dh),3);
% Decimating to outfs
if(infs~=outfs)
dh2 = resample(dh2',outfs,fs)';
end
timeDown = linspace(tw(1),tw(2),size(dh2,2));
eTime = timeDown>=etw(1)&timeDown<=etw(2);
% Normalization
if(~isempty(normFactor))
%dh2 = mapstd(dh2(:,eTime));
    if(normType==1)
    dh2 = (dh2(:,eTime)-normFactor(:,1))./normFactor(:,2);
    end
    if(normType==2)
        dh2 = dh2(:,eTime)./normFactor(:,1);
    end
else
    dh2 = dh2(:,eTime);
end


