function [phBreak,phTime,micSamp,fsAudio] = phonemeExtract(trialInfoMat,fs,wordPhonemeRaw,nonwordPhonemeRaw)
%trialMetrics = [Trials.StartCode];
% trialInfoMat = cell2mat(trialInfo);
 trialNames = extractfield(trialInfoMat,'sound');
 phoneme = [wordPhonemeRaw; nonwordPhonemeRaw];
 phTime = zeros(length(trialInfoMat),6);
 audioFilePath = 'C:\Users\sd355\Box Sync\CoganLab\acoustic_phoneme\';
 audioFiles = [dir(fullfile(audioFilePath,'words\*.wav')) ;dir(fullfile(audioFilePath,'nonwords\*.wav'))] ;
  for i = 1 : length(trialNames)
      i
    trialNames{i} = {trialNames{i}(1:end-4)};
    if(strcmp(trialNames{i},'casif'))
        trialNames{i} = 'casef';
    end
    if(strcmp(trialNames{i},'valek'))
        trialNames{i} = 'valuk';
    end
  %  trialNames{i}    
    p_ind = find(strcmp(phoneme(:,1),(trialNames{i})));
    
    phBreak(i,:) = phoneme(p_ind,2:6);
    phTime(i,:) = (cell2mat(phoneme(p_ind,8:13)));
    strcat(audioFiles(p_ind).folder,'/',audioFiles(p_ind).name)
   [micClean,fsAudio] = audioread(strcat(audioFiles(p_ind).folder,'/',audioFiles(p_ind).name));
   micSamp{i} = micClean;
  end
  phTime = round(phTime*fs/1000);
end