function [ieegall, microphone, trigger] = intanWrap(path, channel, fileNum, isDecimate, isTask)
    % Reads Intan RHD2000 data files and returns the EEG, microphone, and trigger data.
    %
    % Arguments:
    % - path: File path of the Intan data files.
    % - channel: Channel number to extract data from.
    % - fileNum: File number to process.
    % - isDecimate: Logical value indicating whether to decimate the data (default: false).
    % - isTask: Logical value indicating whether to extract task-related data (default: false).
    %
    % Returns:
    % - ieegall: EEG data matrix.
    % - microphone: Microphone data vector.
    % - trigger: Trigger data vector.

    arguments
        path {mustBeTextScalar}
        channel double
        fileNum double
        isDecimate logical = 0
        isTask logical = 0
    end
    
    d = dir([path '*.rhd']);
    d.name
    ieegall = [];
    microphone = [];
    trigger = [];
    
    for iFile = fileNum
        disp(strcat('Recording ', num2str(iFile)));
        fullPathToFile = [path d(iFile).name];
        ieegsamp = [];
        
        [amplifier_data] = read_Intan_RHD2000_file_path_update(path, iFile, channel);
        
        if(isDecimate)
            ieegsamp = (resample(amplifier_data', 2000, 20000))';
        else
            ieegsamp = amplifier_data;
        end
        
        ieegall = [ieegall ieegsamp];
        
        if(isTask)
            [~, board_adc_data] = read_Intan_RHD2000_file_path_update_EDIT_ELIM_OVERHEAD(fullPathToFile, channel(1));
            trigger = [trigger board_adc_data(1, :)];
            microphone = [microphone board_adc_data(2, :)];
        end
    end
end

