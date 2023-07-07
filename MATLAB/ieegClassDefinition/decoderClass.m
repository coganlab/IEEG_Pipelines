classdef decoderClass
    % Class for decoder operations
    
    properties
        numFold double = 20 % Number of folds for cross-validation
        varExplained (1,:) double = 80 % Variance explained threshold
        nIter double = 3 % Number of iterations
    end
    
    methods
        function obj = decoderClass(numFold, varExplained, nIter)
            % Class constructor
            % Creates an instance of the decoderClass
            %   numFold: Number of folds for cross-validation
            %   varExplained: Variance explained threshold
            %   nIter: Number of iterations
            
            obj.numFold = numFold;
            obj.varExplained = varExplained;      
            obj.nIter = nIter;     
        end
        
        function decompStruct = tsneDecompose(obj, ieegStruct, decoderUnit, chanMap, d_time_window, nElec, nIter)
            % Performs t-SNE decomposition
            %   ieegStruct: iEEG class object
            %   decoderUnit: Decoder labels
            %   chanMap: 2D channel map
            %   d_time_window: Decoder time window
            %   nElec: Number of electrodes for analysis
            %   nIter: Number of iterations for t-SNE
            %   Returns decompStruct with t-SNE scores and ratios
            
            arguments
                obj {mustBeA(obj, 'decoderClass')} % Decoder class object
                ieegStruct {mustBeA(ieegStruct, 'ieegStructClass')} % iEEG class object
                decoderUnit double {mustBeVector} % Decoder labels
                chanMap double % 2D channel map
                d_time_window double = ieegStruct.tw % Decoder time window (defaults to epoch time-window)
                nElec double = size(ieegStruct.data, 1) % Number of electrodes for analysis (defaults to total)
                nIter double = 50 % Number of iterations for t-SNE (defaults to 50)
            end
            
            assert(size(ieegStruct.data, 2) == length(decoderUnit), 'Input/output trial mismatch');
            
            [decompStruct.silScore, decompStruct.silRatio] = tsneScoreExtract(ieegStruct.data, decoderUnit, ieegStruct.tw, chanMap, d_time_window, obj.varExplained, nElec, nIter);
        end
        
        function decodeResultStruct = baseClassify(obj, ieegStruct, decoderUnit, options)
            % Performs base classification
            %   obj: decoderClass object
            %   ieegStruct: ieegStructClass object
            %   decoderUnit: Decoder labels
            %   options: Optional arguments for classification
            %       options.d_time_window: Decoder time window (defaults to epoch time-window)
            %       options.selectChannel: Select number of electrodes for analysis (defaults to all)
            %       options.selectTrial: Select number of trials for analysis (defaults to all)
            %       options.isAuc: Select for AUC metric (defaults to 0)
            %   Returns decodeResultStruct with classification results
            
            arguments
                obj {mustBeA(obj, 'decoderClass')} % Decoder class object
                ieegStruct {mustBeA(ieegStruct, 'ieegStructClass')} % ieeg class object
                decoderUnit double {mustBeVector} % Decoder labels
                options.d_time_window double = ieegStruct.tw; % Decoder time window; Defaults to epoch time-window
                options.selectChannel double = 1:size(ieegStruct.data,1); % Select number of electrodes for analysis; Defaults to all
                options.selectTrial double = 1:size(ieegStruct.data,2); % Select number of trials for analysis; Defaults to all
                options.isAuc logical = 0; % Select for AUC metric. Defaults to 0
            end
            
            % Extract values from options
            d_time_window = options.d_time_window;
            selectChannel = options.selectChannel;
            selectTrial = options.selectTrial;
            isAuc = options.isAuc;
            trainTestDiff = 0;
            
            % Check if d_time_window has size 4
            if(sum(size(d_time_window)) == 4)
                trainTestDiff = 1;
            end
            
            % Check for input/output trial mismatch
            assert(size(ieegStruct.data,2) == length(decoderUnit), 'Input/output trial mismatch');
            
            % Select ieegInput based on specified channels and trials
            ieegInput = ieegStruct.data(selectChannel, selectTrial, :);
            decoderUnit = decoderUnit(selectTrial);
            
            % Initialize decodeResultStruct fields
            decodeResultStruct.accPhoneme = 0;
            decodeResultStruct.accPhonemeUnBias = 0;
            
            % Find unique decoder units
            decodeUnitUnique = unique(decoderUnit);
            decodeResultStruct.cmat = zeros(1, length(decodeUnitUnique), length(decodeUnitUnique));
            
            % Initialize variables
            CmatAll = zeros(length(decodeUnitUnique), length(decodeUnitUnique));
            ytestall = [];
            
            % Performclassification for nIter iterations
            for iTer = 1:obj.nIter
                if(trainTestDiff == 0)
                    % Call pcaLinearDecoderWrap function for classification
                    [~, ytest, ypred, optimVarAll, ~, modelWeightsAll] = pcaLinearDecoderWrap(ieegInput, decoderUnit, ieegStruct.tw, d_time_window, obj.varExplained, obj.numFold, isAuc);
                    %[~, ytest, ypred] = stmfDecodeWrap(ieegInput, decoderUnit, ieegStruct.tw, d_time_window, obj.numFold, isauc);
                else
                    % Call pcaLinearDecoderWrapTrainTest function for classification with separate train and test time windows
                    [~, ytest, ypred, optimVarAll, ~, modelWeightsAll] = pcaLinearDecoderWrapTrainTest(ieegInput, decoderUnit, ieegStruct.tw, d_time_window(1,:), d_time_window(2,:), obj.varExplained, obj.numFold, isAuc);
                end
                
                % Accumulate test labels and predictions
                ytestall = [ytestall ytest];
                
                % Compute confusion matrix and accumulate
                Cmat = confusionmat(ytest, ypred);
                CmatAll = CmatAll + Cmat;
            end
            
            % Compute normalized confusion matrix
            CmatCatNorm = CmatAll ./ sum(CmatAll, 2);
            
            % Compute accuracy metrics
            decodeResultStruct.accPhonemeUnBias = trace(CmatAll) / sum(CmatAll(:));
            decodeResultStruct.accPhoneme = trace(CmatCatNorm) / size(CmatCatNorm, 1);
            
            % Store confusion matrix and other results in decodeResultStruct
            decodeResultStruct.cmat = CmatCatNorm;
            decodeResultStruct.p = StatThInv(ytestall, decodeResultStruct.accPhoneme * 100);
            decodeResultStruct.modelWeights = modelWeightsAll;
            decodeResultStruct.optimVarAll = optimVarAll;
        end

        function decodeResultStruct = baseRegress(obj, ieegStruct, decoderUnit, d_time_window, selectChannel, selectTrial)
            % Performs base regression
            %   obj: decoderClass object
            %   ieegStruct: ieegStructClass object
            %   decoderUnit: Decoder labels
            %   d_time_window: Decoder time window (defaults to epoch time-window)
            %   selectChannel: Select number of electrodes for analysis (defaults to all)
            %   selectTrial: Select number of trials for analysis (defaults to all)
            %   Returns decodeResultStruct with regression results
            
            arguments
                obj {mustBeA(obj, 'decoderClass')} % Decoder class object
                ieegStruct {mustBeA(ieegStruct, 'ieegStructClass')} % ieeg class object
                decoderUnit double {mustBeVector} % Decoder labels
                d_time_window double = ieegStruct.tw; % Decoder time window; Defaults to epoch time-window
                selectChannel double = 1:size(ieegStruct.data,1); % Select number of electrodes for analysis; Defaults to all
                selectTrial double = 1:size(ieegStruct.data,2); % Select number of trials for analysis; Defaults to all
            end
            
            % Check if d_time_window has size 4
            trainTestDiff = 0;
            if(sum(size(d_time_window)) == 4)
                trainTestDiff = 1;
            end
            
            % Check for input/output trial mismatch
            assert(size(ieegStruct.data, 2) == length(decoderUnit), 'Input/output trial mismatch');
            
            % Select ieegInput based on specified channels and trials
            ieegInput = ieegStruct.data(selectChannel, selectTrial, :);
            decoderUnit = decoderUnit(selectTrial);
        
            % Perform regression based on train/test differentiation
            if(trainTestDiff == 0)
                % Call pcaLinearRegressDecoderWrap function for regression
                [ytestAll, ypredAll] = pcaLinearRegressDecoderWrap(ieegInput, decoderUnit, ieegStruct.tw, d_time_window, obj.varExplained, obj.numFold);
            else
                % Call pcaLinearRegressDecoderWrapTrainTest function for regression with separate train and test time windows
                [ytestAll, ypredAll] = pcaLinearRegressDecoderWrapTrainTest(ieegInput, decoderUnit, ieegStruct.tw, d_time_window(1,:), d_time_window(2,:), obj.varExplained, obj.numFold);
            end
            
            % Fit linear regression model and compute R-squared and p-value
            distMod = fitlm(ytestAll, ypredAll);
            r2 = distMod.Rsquared.Ordinary;
            pVal = distMod.Coefficients.pValue(2);
            
            % Store regression results in decodeResultStruct
            decodeResultStruct.r2 = r2;
            decodeResultStruct.pVal = pVal;
        end

        function decodeTimeStruct = tempGenClassify1D(obj, ieegStruct, decoderUnit, options)
            % Generates temporal classification results
            %   obj: decoderClass object
            %   ieegStruct: ieegStructClass object
            %   decoderUnit: Decoder labels
            %   options.timeRes: Decoder time resolution (defaults to 0.02)
            %   options.timeWin: Time window size for analysis
            %   options.selectChannels: Select number of electrodes for analysis (defaults to all)
            %   options.selectTrials: Select number of trials for analysis (defaults to all)
            %   Returns decodeTimeStruct with temporal classification results
            
            arguments
                obj {mustBeA(obj, 'decoderClass')} % Decoder class object
                ieegStruct {mustBeA(ieegStruct, 'ieegStructClass')} % ieeg class object
                decoderUnit double {mustBeVector} % Decoder labels
                options.timeRes double = 0.02; % Decoder time resolution; Defaults to 0.02
                options.timeWin double; % Time window size for analysis
                options.selectChannels double = 1:size(ieegStruct.data,1); % Select number of electrodes for analysis; Defaults to all
                options.selectTrials double = 1:size(ieegStruct.data,2); % Select number of trials for analysis; Defaults to all
                options.isModelWeight logical  = 1 % Extract model weights if true;
            end
            
            % Retrieve options values
            timeRes = options.timeRes;
            timeWin = options.timeWin;
            selectChannels = options.selectChannels;
            selectTrials = options.selectTrials;
            
            % Define time range based on ieegStruct.tw, timeRes, and timeWin
            timeRange = ieegStruct.tw(1):timeRes:ieegStruct.tw(2)-timeWin;
            
            % Initialize arrays for storing results
            accTime = zeros(1, length(timeRange));
            pValTime = zeros(1, length(timeRange));
            nTime = length(timeRange);
            modelweightTime = cell(1, length(timeRange));
            
            % Perform temporal classification in parallel for each time point
            parfor iTime = 1:nTime
                % Call baseClassify function for classification at each time point
                decodeResultStruct = baseClassify(obj, ieegStruct, decoderUnit, d_time_window =  [timeRange(iTime) timeRange(iTime)+timeWin], selectChannel =  selectChannels, selectTrial =  selectTrials);
                
                % Store accuracy and p-value at each time point
                accTime(iTime) = decodeResultStruct.accPhoneme;
                pValTime(iTime) = decodeResultStruct.p;
                
                % Extract model weights for each time point
                modelweightTime{iTime} = extractPcaLdaModelWeights(decodeResultStruct.modelWeights, size(decodeResultStruct.cmat, 1), length(selectChannels), timeWin * ieegStruct.fs);
            end
            
            % Store temporal classification results in decodeTimeStruct
            decodeTimeStruct.accTime = accTime;
            decodeTimeStruct.timeRange = timeRange;
            decodeTimeStruct.pValTime = pValTime;
            if(options.isModelWeight)
                decodeTimeStruct.modelweightTime = modelweightTime;
            end
        end

        function decodeTimeStruct = tempGenClassify2D(obj, ieegStruct, decoderUnit, options)
            % Generates 2D temporal classification results
            %   obj: decoderClass object
            %   ieegStruct: ieegStructClass object
            %   decoderUnit: Decoder labels
            %   options.timeRes: Decoder time resolution (defaults to 0.02)
            %   options.timeWin: Time window size for analysis (defaults to 0.2)
            %   options.selectChannels: Select number of electrodes for analysis (defaults to all)
            %   options.selectTrials: Select number of trials for analysis (defaults to all)
            %   Returns decodeTimeStruct with 2D temporal classification results
            
            arguments
                obj {mustBeA(obj, 'decoderClass')} % Decoder class object
                ieegStruct {mustBeA(ieegStruct, 'ieegStructClass')} % ieeg class object
                decoderUnit double {mustBeVector} % Decoder labels
                options.timeRes double = 0.02; % Decoder time resolution; Defaults to 0.02
                options.timeWin double = 0.2; % Time window size for analysis; Defaults to 0.2
                options.selectChannels double = 1:size(ieegStruct.data,1); % Select number of electrodes for analysis; Defaults to all
                options.selectTrials double = 1:size(ieegStruct.data,2); % Select number of trials for analysis; Defaults to all
            end
            
            % Retrieve options values
            timeRes = options.timeRes;
            timeWin = options.timeWin;
            selectChannels = options.selectChannels;
            selectTrials = options.selectTrials;
            
            % Define time range based on ieegStruct.tw, timeRes, and timeWin
            timeRange = ieegStruct.tw(1):timeRes:ieegStruct.tw(2)-timeWin;
            
            % Initialize arrays for storing results
            accTime = zeros(length(timeRange), length(timeRange));
            pValTime = zeros(length(timeRange), length(timeRange));
            nTime = length(timeRange);
            
            % Perform 2D temporal classification in parallel for each time point
            parfor iTimeTrain = 1:nTime
                tsTrain = [timeRange(iTimeTrain) timeRange(iTimeTrain)+timeWin];
                for iTimeTest = 1:nTime        
                    tsTest = [timeRange(iTimeTest) timeRange(iTimeTest)+timeWin];
                    % Call baseClassify function for classification at each time point
                    decodeResultStruct = baseClassify(obj, ieegStruct, decoderUnit, d_time_window = [tsTrain; tsTest], selectChannel = selectChannels, selectTrial = selectTrials);
                    
                    % Store accuracy and p-value at each time point
                    accTime(iTimeTrain, iTimeTest) = decodeResultStruct.accPhoneme;
                    pValTime(iTimeTrain, iTimeTest) = decodeResultStruct.p;
                end
            end
            
            % Store 2D temporal classification results in decodeTimeStruct
            decodeTimeStruct.accTime = accTime;
            decodeTimeStruct.timeRange = timeRange;
            decodeTimeStruct.pValTime = pValTime;
        end
           
        function decodeTimeStruct = tempGenRegress1D(obj, ieegStruct, decoderUnit, options)
            % Generates 1D temporal regression results
            %   obj: decoderClass object
            %   ieegStruct: ieegStructClass object
            %   decoderUnit: Decoder labels
            %   options.timeRes: Decoder time resolution (defaults to 0.02)
            %   options.timeWin: Time window size for analysis (defaults to 0.2)
            %   options.selectChannels: Select number of electrodes for analysis (defaults to all)
            %   options.selectTrials: Select number of trials for analysis (defaults to all)
            %   Returns decodeTimeStruct with 1D temporal regression results
            
            arguments
                obj {mustBeA(obj, 'decoderClass')} % decoder class object
                ieegStruct {mustBeA(ieegStruct, 'ieegStructClass')} % ieeg class object
                decoderUnit double {mustBeVector} % Decoder labels
                options.timeRes double = 0.02; % decoder time resolution; Defaults to 0.02
                options.timeWin double = 0.2; % Time window size for analysis; Defaults to 0.2
                options.selectChannels double = 1:size(ieegStruct.data,1); % select number of electrodes for analysis; Defaults to all
                options.selectTrials double = 1:size(ieegStruct.data,2); % select number of trials for analysis; Defaults to all
            end
            
            % Retrieve options values
            timeRes = options.timeRes;
            timeWin = options.timeWin;
            selectChannels = options.selectChannels;
            selectTrials = options.selectTrials;
            
            % Define time range based on ieegStruct.tw, timeRes, and timeWin
            timeRange = ieegStruct.tw(1):timeRes:ieegStruct.tw(2)-timeWin;
            
            % Preallocate arrays for storing results
            nTime = length(timeRange);
            r2Time = zeros(1, nTime);
            pValTime = nan(1, nTime);
            
            % Perform 1D temporal regression in parallel for each time point
            parfor iTime = 1:nTime
                % Call baseRegress function for regression at each time point
                decodeResultStruct = baseRegress(obj, ieegStruct, decoderUnit, [timeRange(iTime) timeRange(iTime)+timeWin], selectChannels, selectTrials);
                
                % Store R-squared and p-value in the corresponding position
                r2Time(iTime) = decodeResultStruct.r2;
                pValTime(iTime) = decodeResultStruct.pVal;
            end
            
            % Store 1D temporal regression results in decodeTimeStruct
            decodeTimeStruct.r2Time = r2Time;
            decodeTimeStruct.pValTime = pValTime;
            decodeTimeStruct.timeRange = timeRange;
        end

        function decodeTimeStruct = tempGenRegress2D(obj,ieegStruct,decoderUnit,options)
            arguments
                obj {mustBeA(obj,'decoderClass')} % decoder class object
                ieegStruct {mustBeA(ieegStruct,'ieegStructClass')} % ieeg class object
                decoderUnit double {mustBeVector} % Decoder labels
                options.timeRes double = 0.02; % decoder time resolution; Defaults to 0.02
                options.timeWin double = 0.2;
                options.selectChannels double = 1:size(ieegStruct.data,1); % select number of electrodes for analysis; Defaults to all
                options.selectTrials double = 1:size(ieegStruct.data,2); % select number of trials for analysis; Defaults to all
               
             end
            timeRes = options.timeRes;
            timeWin = options.timeWin;
            selectChannels = options.selectChannels;
            selectTrials = options.selectTrials;
            timeRange = ieegStruct.tw(1):timeRes:ieegStruct.tw(2)-timeWin;
            nTime = length(timeRange);
            r2Time = zeros(length(timeRange),length(timeRange));
            pValTime =nan(length(timeRange),length(timeRange));
            %f = waitbar(0, 'Starting');
           
            
            parfor iTimeTrain = 1:nTime   
                tsTrain = [timeRange(iTimeTrain) timeRange(iTimeTrain)+timeWin];
                for iTimeTest = 1:nTime        
                    tsTest = [timeRange(iTimeTest) timeRange(iTimeTest)+timeWin];
                    decodeResultStruct = baseRegress(obj,ieegStruct,decoderUnit,[tsTrain; tsTest],selectChannels,selectTrials);
                    r2Time(iTimeTrain,iTimeTest) = decodeResultStruct.r2;
                    pValTime(iTimeTrain,iTimeTest) = decodeResultStruct.pVal;                    
                end
                %waitbar(iTimeTrain/length(timeRange), f, sprintf('Progress: %d %%', floor(iTimeTrain/length(timeRange)*100)));
            end
            %close(f);
            decodeTimeStruct.r2Time = r2Time;
            decodeTimeStruct.pValTime = pValTime;
            decodeTimeStruct.timeRange = timeRange;
        end
        
    end
end