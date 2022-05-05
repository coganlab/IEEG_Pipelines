function filtSig = getFilterSignal(ieeg,fs,fband)

pfilt=designfilt('bandpassiir',...       % band pass filter for beta extraction
    'FilterOrder',8,'PassbandFrequency1',fband(1),'PassbandFrequency2',fband(2),...
    'PassbandRipple',0.2,'SampleRate',fs);


for iChan = 1:size(ieeg,1)
    
        filtSig(iChan,:) = filtfilt(pfilt,ieeg(iChan,:));
        
end
end