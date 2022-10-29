classdef ieegStructClass
    
    properties 
        data % channels x trials x timepoints
        fs % sampling frequency
        tw % time-epoch
        fBand % frequency window;   
        name % Epoch name
    end
    
    methods
        function obj = ieegStructClass(data, fs, tw, fBand, name)% class constructor
            
            obj.data = data;
            obj.fs = fs;
            obj.tw = tw;
            obj.fBand = fBand;        
            obj.name = name;
        end        
        function objCar = extractCar(obj,badChannels)% common average referencing
            if (nargin<2)
                badChannels = [];
            end
            disp(['Common average filtering ' obj.name]);
            objCar = obj;
            
            objCar.data = carFilterImpedance(obj.data,badChannels); 
            objCar.name = strcat(obj.name, '_CAR');
        end % Common average referencing        
        function [ieegHiGamma,ieegHiGammaPower] = extractHiGamma(obj,fDown,gtw,normFactor,normType) % Extracting high-gamma signal           
            % Extracts HG of ieeg (active) with normalization factors
            % if provided
            % Input:
            % obj - ieeg structure for baseline
            % fDown - Downsampled frequency (Optional; if not present use
            % same sampling frequency)
            % gtw - gamma time window to normalize; (Optional; if not
            % present use ieeg time-epoch
            % normFactor - normalization values (channels x 2; if not
            % present, no normalization
            % 
            % Output:
            % ieegHiGammaNorm - Extracted HG structure;
            
            disp(['Extracting High Gamma ' obj.name]);
            fGamma = [70 150];
            dataTemp = obj.data;
            fsTemp = obj.fs;
            twTemp = obj.tw;
            ieegGammaTemp = [];
            
            
            switch nargin
                case 1
                    fDown = fsTemp;
                    gtw = twTemp;
                    normFactor = [];
                    normType = 1;
                    obj.name = strcat(obj.name , '_High-Gamma');
                case 2
                    gtw = twTemp;
                    normFactor = [];   
                    normType = 1;
                    obj.name = strcat(obj.name , '_High-Gamma');
                case 3                   
                    normFactor = [];  
                    normType = 1;
                    obj.name = strcat(obj.name , '_High-Gamma');
                case 4
                    normType = 1;
                    obj.name = strcat(obj.name , '_High-Gamma-Normalized');
                case 5                    
                    obj.name = strcat(obj.name , '_High-Gamma-Normalized');
            end
            
        % % no idea how to fix this so this is the best I got
        % chanFlag=0;
            if size(dataTemp,1)==1
                [~,ieegGammaTemp(1,:,:)] = EcogExtractHighGammaTrial(double(squeeze(dataTemp)),fsTemp,fDown,fGamma,twTemp,gtw,normFactor,normType); 
                
            else
                for iTrial = 1:size(obj.data,2)    
                    [~,ieegGammaTemp(:,iTrial,:)] = EcogExtractHighGammaTrial(double(squeeze(dataTemp(:,iTrial,:))),fsTemp,fDown,fGamma,twTemp,gtw,normFactor,normType);
                
                end
            end
%             if chanFlag==1
%                 ieegGammaTemp=ieegGammaTemp(1,:,:);
%             end
            
            ieegHiGamma = ieegStructClass(ieegGammaTemp, fDown, gtw, fGamma,obj.name);
            ieegHiGammaPower = (squeeze(mean(log10(ieegGammaTemp.^2),3)));
        end

        function normFactor = extractHGnormFactor(obj) 
            % Extracts normalization factors for ieeg; mean & standard
            % deviation
             normFactor = zeros(size(obj.data,1),2);
            parfor iChan = 1:size(obj.data,1)
                normFactor(iChan,:) = [mean2(squeeze(obj.data(iChan,:,:))) std2(squeeze(obj.data(iChan,:,:)))];
            end
        end % Extracting HG normalization factors    

        function [ieegHiGammaNorm, normFactor] = extractHiGammaNorm(obj1,obj2,fDown,gtw1,gtw2) % extracting normalized high-gamma
            % Extracts normalized HG of ieeg1 (active) with normalization factors
            % from ieeg2 (passive)
            % Input:
            % obj1 - active ieeg structure for target (auditory, go,response)
            % obj2 - passive ieeg structure for baseline 
            % fDown - Downsampled frequency (optional)
            % gtw1 - output time window after normalization; (optional)
            % gtw2 - base time window to normalize (optional)
            
            % Output:
            % ieegHiGammaNorm - Normalized HG structure;
            switch nargin
                case 2
                    assert(obj1.fs==obj2.fs,'Sampling Frequency mismatch');
                    fDown = obj1.fs;
                    gtw1 = obj1.tw;
                    gtw2 = obj2.tw;
                case 3
                    gtw1 = obj1.tw;
                    gtw2 = obj2.tw;                  
                case 4                   
                    gtw2 = obj2.tw;              
                    
            end 
            ieegHiGammaBase = extractHiGamma(obj2,fDown,gtw2);
            normFactor = extractHGnormFactor(ieegHiGammaBase);            
            ieegHiGammaNorm = extractHiGamma(obj1,fDown,gtw1,normFactor);            
        end
        function ieegHiGammaNorm = normHiGamma(obj1,normFactor, normType)
            ieegHiGammaNormData = obj1.data;
            parfor iChan = 1:size(obj1.data,1)
                if(normType==1)
                    ieegHiGammaNormData(iChan,:,:) = (obj1.data(iChan,:,:)-normFactor(iChan,1))./normFactor(iChan,2);
                end
                if(normType==2)
                    ieegHiGammaNormData(iChan,:,:) = (obj1.data(iChan,:,:)-normFactor(iChan,1));
                end
            end
            ieegHiGammaNorm = obj1;
            ieegHiGammaNorm.data = ieegHiGammaNormData;
        end
        function chanSig = extractTimePermCluster(obj1,obj2)
            % Input
            % obj1 - active IEEG structure
            % obj2 - passive IEEG structure
            % Output
            % chanSig - cluster correction output
            chanSig = {};
            baseData = obj2.data;
            targetData = obj1.data;      
            time2pad = size(targetData,3)/size(baseData,3);            
            parfor iChan = 1:size(baseData,1)
                 % Assumption - target data window is longer than base data
                % Correction - Random sampling & padding base window trials
                % to account for time difference
                baseDataChan = squeeze(baseData(iChan,:,:));
                targetDataChan = squeeze(targetData(iChan,:,:));
                baseDataChanPad = zeros(size(targetDataChan));
                
                for iTrial = 1:size(baseDataChan,1)
                    randTrials = datasample(1:size(baseDataChan,1),time2pad-1,'Replace',false);
                    trials2join = baseDataChan(randTrials,:);
                    baseDataChanPad(iTrial,:) = [baseDataChan(iTrial,:) trials2join(:)'];
                end
                [zValsRawAct, pValsRaw, actClust]=timePermCluster(targetDataChan,baseDataChanPad,1000,1,1.645);
                chanSig{iChan}.zValsRawAct=zValsRawAct;
                chanSig{iChan}.pValsRaw=pValsRaw;
                chanSig{iChan}.actClust=actClust;               
                disp(iChan)
            end
            
        end % Time Series permutation cluster          
        
        
    end
end