classdef ieegStructClass
    % The ieegStructClass is a class for handling iEEG data with various operations.
    % It provides methods for common average referencing, band-pass filtering, high-gamma extraction,
    % normalization, permutation cluster analysis, and more.
    
    properties 
        data % channels x trials x timepoints
        fs % sampling frequency
        tw % time-epoch
        fBand % frequency window   
        name % Epoch name
    end
    
    methods
        function obj = ieegStructClass(data, fs, tw, fBand, name)
            % Class constructor
            
            % Initialize properties
            obj.data = data;
            obj.fs = fs;
            obj.tw = tw;
            obj.fBand = fBand;        
            obj.name = name;
        end
        function objCar = extractCableCar(obj, channelNames)
            % Common average referencing
            % Extracts CAR-filtered data by subtracting the common average across channels
           assert(length(channelNames)==size(obj.data,1), 'Channel dimension mismatch')
           for iChan = 1:length(channelNames)
               chanNameShank{iChan} = ieegChanLabelParse(channelNames{iChan});
           end
           if(cellfun(@isempty,chanNameShank))
                chanNameShank{cellfun(@isempty,chanNameShank)} = 'noName';
           end
           [chanNameUnique,~,chanId] = unique(chanNameShank,'stable');
            
            disp(['Common average filtering across cable' obj.name]);
            objCar = obj;
            
            % Apply common average referencing
            carFilt = carFilterImpedance(obj.data, []); 
            % Apply cable average referencing
            for iChan = 1:length(chanNameUnique)
                disp(['Cable average filtering ' chanNameUnique{iChan}]);
                chan2average = chanId==iChan;
                if(sum(chan2average)>1)
                    objCar.data(chan2average,:,:) = carFilter(obj.data(chan2average,:,:)); 
                else
                    objCar.data(chan2average,:,:) = carFilt(chan2average,:,:);
                end
            end
            disp(['Nan channels ' sum(isnan(objCar.data(:,1,1)))]);
            objCar.name = strcat(obj.name, '_CAR_cable');
        end
        
        function objCar = extractCar(obj, badChannels)
            % Common average referencing
            % Extracts CAR-filtered data by subtracting the common average across channels
            
            if nargin < 2
                badChannels = [];
            end
            
            disp(['Common average filtering ' obj.name]);
            objCar = obj;
            
            % Apply common average referencing
            objCar.data = carFilterImpedance(obj.data, badChannels); 
            objCar.name = strcat(obj.name, '_CAR');
        end
        
        function [ieegFilter, ieegPower] = extractBandPassFilter(obj, fBand, fDown, gtw)
            % Extract band-pass filtered signal
            
            arguments
                obj ieegStructClass
                fBand double 
                fDown double = obj.fs;
                gtw double = obj.tw;
            end
            
            dataTemp = obj.data;
            
            if size(dataTemp, 1) == 1
                ieegFilterTemp(1, :, :) = ExtractLowFrequencyWrap(dataTemp, obj.fs, fDown, fBand, obj.tw, gtw);
            else
                for iTrial = 1:size(obj.data, 2)    
                    ieegFilterTemp(:, iTrial, :) = ExtractLowFrequencyWrap(squeeze(dataTemp(:, iTrial, :)), obj.fs, fDown, fBand, obj.tw, gtw);
                end
            end
            
            obj.name = strcat(obj.name, '_band-pass_filtered');
            
            ieegFilter = ieegStructClass(ieegFilterTemp, fDown, gtw, fBand, obj.name);
            ieegPower = squeeze(mean(log10(ieegFilterTemp.^2), 3));
        end
        
        function [ieegHiGamma, ieegHiGammaPower] = extractHiGamma(obj, fDown, gtw, normFactor, normType)
            % Extract high-gamma signal
            
            % Extracts high-gamma signal from the iEEG data
            % Input:
            %   fDown: Downsampled frequency (Optional; if not present, use the same sampling frequency)
            %   gtw: Gamma time window to normalize (Optional; if not present, use the iEEG time-epoch)
            %   normFactor: Normalization values (channels x 2; if not present, no normalization)
            %   normType: Normalization type (1: z-score normalization, 2: mean subtraction) (Optional; default: 1)
            % 
            % Output:
            %   ieegHiGamma: Extracted high-gamma structure
            %   ieegHiGammaPower: Power of the extracted high-gamma signal
            
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
                    obj.name = strcat(obj.name, '_High-Gamma');
                case 2
                    gtw = twTemp;
                    normFactor = [];   
                    normType = 1;
                    obj.name = strcat(obj.name, '_High-Gamma');
                case 3                   
                    normFactor = [];  
                    normType = 1;
                    obj.name = strcat(obj.name, '_High-Gamma');
                case 4
                    normType = 1;
                    obj.name = strcat(obj.name, '_High-Gamma-z-score-normalized');
                case 5 
                    switch(normType)
                        case 1
                            obj.name = strcat(obj.name, '_High-Gamma-z-score-normalized');
                        case 2
                            obj.name = strcat(obj.name, '_High-Gamma-mean-subtracted-normalized'); 
                        case 3
                            obj.name = strcat(obj.name, '_High-Gamma-abs-rel-baseline-normalized');
                        case 4
                            obj.name = strcat(obj.name, '_High-Gamma-perc-ratio-baseline-normalized');
                        case 5
                            obj.name = strcat(obj.name, '_High-Gamma-log-baseline-normalized (unit dB)');
                        case 6
                            obj.name = strcat(obj.name, '_High-Gamma-norm-baseline-normalized');
                    end
            end
            isPower = 1;
            if size(dataTemp, 1) == 1
                [~, ieegGammaTemp(1, :, :)] = EcogExtractHighGammaTrial(double(squeeze(dataTemp)), fsTemp, fDown, fGamma, twTemp, gtw, normFactor, normType,isPower); 
            else
                for iTrial = 1:size(obj.data, 2)    
                    [~, ieegGammaTemp(:, iTrial, :)] = EcogExtractHighGammaTrial(double(squeeze(dataTemp(:, iTrial, :))), fsTemp, fDown, fGamma, twTemp, gtw, normFactor, normType,isPower);
                end
            end
            
            ieegHiGamma = ieegStructClass(ieegGammaTemp, fDown, gtw, fGamma, obj.name);
            ieegHiGammaPower = squeeze(mean(log10(ieegGammaTemp), 3));
        end

        function normFactor = extractHGnormFactor(obj)
            % Extract normalization factors for ieeg (mean & standard deviation)
            
            % Calculates the mean and standard deviation normalization factors for each channel in the iEEG data
           % [NumTrials, goodtrials] = remove_bad_trials(obj.data, 10);
            normFactor = zeros(size(obj.data, 1), 2);
            for iChan = 1:size(obj.data, 1)
               % normFactor(iChan, :) = [mean2(squeeze(obj.data(iChan, goodtrials(iChan,:),:)),"omitnan"), std2(squeeze(obj.data(iChan,  goodtrials(iChan,:),:)),"omitnan")];
           normFactor(iChan, :) = [mean(squeeze(obj.data(iChan, :)),"omitnan"), std(squeeze(obj.data(iChan,  :)),"omitnan")];
            end
        end
        
        function [ieegHiGammaNorm, normFactor] = extractHiGammaNorm(obj1, obj2, fDown, gtw1, gtw2)
            % Extract normalized high-gamma
            
            % Extracts normalized high-gamma signal of obj1 (active) with normalization factors from obj2 (passive)
            % Input:
            %   obj1: Active ieegStructClass object for the target (auditory, go, response)
            %   obj2: Passive ieegStructClass object for the baseline 
            %   fDown: Downsampled frequency (optional)
            %   gtw1: Output time window after normalization (optional)
            %   gtw2: Base time window to normalize (optional)
            % 
            % Output:
            %   ieegHiGammaNorm: Normalized high-gamma structure
            
            switch nargin
                case 2
                    assert(obj1.fs == obj2.fs, 'Sampling Frequency mismatch');
                    fDown = obj1.fs;
                    gtw1 = obj1.tw;
                    gtw2 = obj2.tw;
                case 3
                    gtw1 = obj1.tw;
                    gtw2 = obj2.tw;                  
                case 4                   
                    gtw2 = obj2.tw;              
            end 
            
            ieegHiGammaBase = extractHiGamma(obj2, fDown, gtw2);
            normFactor = extractHGnormFactor(ieegHiGammaBase);
            ieegHiGammaNorm = extractHiGamma(obj1, fDown, gtw1, normFactor);
        end
        
        function ieegHiGammaNorm = normHiGamma(obj1, normFactor, normType)
            % Normalize high-gamma
            
            % Normalizes the high-gamma signal using normalization factors
            
            ieegHiGammaNormData = obj1.data;
            
            for iChan = 1:size(obj1.data, 1)
                if normType == 1
                    ieegHiGammaNormData(iChan, :, :) = (obj1.data(iChan, :, :) - normFactor(iChan, 1)) / normFactor(iChan, 2);
                end
                if normType == 2
                    ieegHiGammaNormData(iChan, :, :) = obj1.data(iChan, :, :) - normFactor(iChan, 1);
                end
            end
            
            ieegHiGammaNorm = obj1;
            ieegHiGammaNorm.data = ieegHiGammaNormData;
        end
        
        function chanSig = extractTimePermCluster(obj1, obj2)
            % Time Series permutation cluster
            
            % Performs time series permutation cluster analysis on the iEEG data
            
            % Input:
            %   obj1: Active ieegStructClass object
            %   obj2: Passive ieegStructClass object
            % 
            % Output:
            %   chanSig: Cluster correction output
            
            chanSig = {};
            baseData = obj2.data;
            targetData = obj1.data;
            time2pad = size(targetData, 3) / size(baseData, 3);
            
            parfor iChan = 1:size(baseData, 1)
                % Assumption: target data window is longer than base data
                % Correction: Random sampling & padding base window trials
                % to account for time difference
                
                baseDataChan = squeeze(baseData(iChan, :, :));
                targetDataChan = squeeze(targetData(iChan, :, :));
                baseDataChanPad = zeros(size(targetDataChan));
                
                for iTrial = 1:size(baseDataChan, 1)
                    randTrials = datasample(1:size(baseDataChan, 1), time2pad - 1, 'Replace', false);
                    trials2join = baseDataChan(randTrials, :);
                    baseDataChanPad(iTrial, :) = [baseDataChan(iTrial, :) trials2join(:)'];
                end
                
                [zValsRawAct, pValsRaw, actClust] = timePermCluster(targetDataChan, baseDataChanPad, 1000, 1, 1.645);
                chanSig{iChan}.zValsRawAct = zValsRawAct;
                chanSig{iChan}.pValsRaw = pValsRaw;
                chanSig{iChan}.actClust = actClust;               
                disp(iChan)
            end            
        end     
    end
end
