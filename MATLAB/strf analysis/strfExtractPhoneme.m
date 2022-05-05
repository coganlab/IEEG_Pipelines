function strfExtractPhoneme(audioStimulusPhoneme,pLabel,phonEvok,tw,fs)
 pLabelUnique = unique(pLabel);
 for pU = 1:length(pLabelUnique)
     pTrials = find(pLabel == pLabelUnique(pU));
     ieeg2strf = phonEvok(:,:,pTrials);
     ieeg2strf = permute(ieeg2strf,[1 3 2]);
     audioFiles = [];
     for i = 1:length(pTrials)
        audioFiles{i} = char(get_filenames(audioStimulusPhoneme, strcat('soundTrial',num2str(pTrials(i)),'.wav'), 1));
     end
     audioFiles = audioFiles';
     preprocStimParams = struct;      %create preprocessing param structure
     preprocStimParams.tfType = 'stft'; %use short-time FT
     tfParams = struct;               %create time-frequency params
     tfParams.high_freq = 8000;       %specify max freq to analyze
     tfParams.low_freq = 1;         %specify min freq to analyze
     tfParams.log = 1;                %take log of spectrogram
     tfParams.dbnoise = 80;           %cutoff in dB for log spectrogram, ignore anything below this
     tfParams.refpow = 0;             %reference power for log spectrogram, set to zero for max of spectrograms across stimuli

     preprocStimParams.tfParams = tfParams;
 
     tempPreprocDir = tempname();    
     [s,mess,messid] = mkdir(tempPreprocDir);
     preprocStimParams.outputDir = tempPreprocDir;
     [wholeStim, groupIndex, stimInfo, preprocStimParams] = preprocSound(audioFiles, preprocStimParams);
     %%
     time = linspace(tw(1),tw(2),size(ieeg2strf,3));
    eTime = time>=0&time<=stimInfo.stimLengths(1);
    parfor i = 1:size(ieeg2strf,1)
        sig2Analyze = double(squeeze(ieeg2strf(i,:,:)));
        %ieegGamma = eegfilt(sig2Analyze,fs,70,150,0,200);
        wholeResp = [];
        for tr =1:size(sig2Analyze,1)  
            %gammaPower = decimate(abs(hilbert(ieegGamma(tr,eTime))),2);  
             %gammaPower =  gammaPower./max(gammaPower);
            timeOriginal = time(eTime);
            timeInterp = linspace(timeOriginal(1),timeOriginal(end),round((timeOriginal(end)-timeOriginal(1))*1000));
            sigInterp = spline(timeOriginal,sig2Analyze(tr,eTime),timeInterp);
            wholeResp = horzcat(wholeResp,sigInterp);
        end
        %wholeResp = zscore(wholeResp);
        strfData(wholeStim,wholeResp,groupIndex);
        strfLength = 300;
        strfDelays = 0:(strfLength-1);
        modelParams = linInit(stimInfo.numStimFeatures, strfDelays);

        trainingGroups = 1:round(length(goodTrialsCommon)*0.6);
        trainingIndex = findIdx(trainingGroups, groupIndex);    
        earlyStoppingGroups = round(length(goodTrialsCommon)*0.6)+1;
        earlyStoppingIndex = findIdx(earlyStoppingGroups, groupIndex);
        optOptions = trnGradDesc();
        optOptions.display = 1;
        optOptions.maxIter = 1000;
        optOptions.stepSize = 0.001;
        optOptions.earlyStop = 1;
        optOptions.gradNorm = 1;
        fprintf('\nRunning Gradient Descent training...\n');
        [modelParamsGD, optOptions] = strfOpt(modelParams, trainingIndex, optOptions, earlyStoppingIndex);
        %strfWeightGD(i,:,:) =  modelParamsGD.w1;
        modelParamsFit{pU,i} = modelParamsGD;
    end
 end
end