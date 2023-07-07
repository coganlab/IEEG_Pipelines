classdef ieegStructMicro < ieegStructClass
    % The ieegStructMicro is a subclass of ieegStructClass specialized for micro-scale iEEG data.
    % It extends the functionality of ieegStructClass by adding additional methods for spatial smoothing
    % and spatial averaging specific to micro-scale data.
    
    properties 
        chanMap % Channel map for micro-scale iEEG data
    end
    
    methods
        function obj = ieegStructMicro(data, fs, tw, fBand, name, chanMap)
            % Class constructor
            
            % Calls the superclass constructor and initializes the properties
            obj@ieegStructClass(data, fs, tw, fBand, name);
            obj.chanMap = chanMap;
        end
        
        function ieegSpatialSmooth = spatialSmoothMicro(obj, window)
            % Spatial smoothing of micro-scale iEEG data
            
            % Applies spatial smoothing to the iEEG data using the channel map and a given window size
            % Input:
            %   window: Window size for spatial smoothing
            % Output:
            %   ieegSpatialSmooth: Spatially smoothed ieegStructMicro object
            
            dataTemp = obj.data;
            chanMapTemp = obj.chanMap;
            dataSmooth = zeros(size(dataTemp));
            
            for iTrial = 1:size(dataTemp, 2)
                dataSmooth(:, iTrial, :) = spatialSmooth(squeeze(dataTemp(:, iTrial, :)), chanMapTemp, window);
            end
            
            ieegSpatialSmooth = ieegStructMicro(dataSmooth, obj.fs, obj.tw, obj.fBand, [obj.name '_' num2str(window(1)) 'x' num2str(window(2)) '_spatially_smoothed'], chanMapTemp);
        end
        
        function [ieegSpatialAverage, matrixPoints] = spatialAverage(obj, window, isOverlap)
            % Spatial averaging of micro-scale iEEG data
            
            % Performs spatial averaging on the iEEG data using the channel map and a given window size
            % Input:
            %   window: Window size for spatial averaging
            %   isOverlap: Flag indicating whether overlapping windows are allowed
            % Output:
            %   ieegSpatialAverage: Spatially averaged ieegStructMicro object
            %   matrixPoints: Indices of the averaged channels in the channel map
            
            matrixPoints = matrixSubSample(obj.chanMap, window, isOverlap);
            chanMapAverage = zeros(floor(size(obj.chanMap, 1) / window(1)), floor(size(obj.chanMap, 2) / window(2)));
            dataAverage = zeros(size(matrixPoints, 1), size(obj.data, 2), size(obj.data, 3));
            
            for iSA = 1:size(matrixPoints, 1)
                dataAverage(iSA, :, :) = mean(obj.data(matrixPoints(iSA, :), :, :), 1);
                chanMapAverage(iSA) = iSA;
            end
            
            ieegSpatialAverage = ieegStructMicro(dataAverage, obj.fs, obj.tw, obj.fBand, [obj.name '_' num2str(window(1)) 'x' num2str(window(2)) '_spatially_averaged'], chanMapAverage);
        end
        
        function ieegHiGammaNorm = extractHiGammaNorm(obj1, obj2, fDown, gtw1, gtw2)
            % Extract normalized high-gamma from micro-scale iEEG data
            
            % Extracts the normalized high-gamma signal from micro-scale iEEG data
            % Input:
            %   obj1: Active ieegStructMicro object for the target (auditory, go, response)
            %   obj2: Passive ieegStructMicro object for the baseline 
            %   fDown: Downsampled frequency (optional)
            %   gtw1: Output time window after normalization (optional)
            %   gtw2: Base time window to normalize (optional)
            % Output:
            %   ieegHiGammaNorm: Normalized high-gamma ieegStructMicro object
            
            ieegHiGammaTemp = extractHiGammaNorm@ieegStructClass(obj1, obj2, fDown, gtw1, gtw2);
            ieegHiGammaNorm = ieegStructMicro(ieegHiGammaTemp.data, ieegHiGammaTemp.fs, ieegHiGammaTemp.tw, ieegHiGammaTemp.fBand, ieegHiGammaTemp.name, obj1.chanMap);
        end
    end
end
