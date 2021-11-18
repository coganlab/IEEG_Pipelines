% top level wrapper for preProcess workflow
%global BOX_DIR
%global RECONDIR
BOX_DIR='C:\Users\gcoga\Box';
RECONDIR='C:\Users\gcoga\Box\ECoG_Recon';
Task=[];
%Task.Name='Phoneme_Sequencing';%'LexicalDecRepDelay';
Task.Name='LexicalDecRepNoDelay';
Task.Outlier.Field='Auditory'; % this is used to define the field for outlier channel removals
preProcess_LineFilter(Task);
preProcess_ChannelOutlierRemoval(Task);
%preProcess_ResponseCoding;
%preProcess_Stats
%preProcess_Specgrams
