function convert_OpenE_rec2mat(fpath)
    % Converts Open Ephys recordings to MATLAB .mat format.
    %
    % Arguments:
    % - fpath: File path of the Open Ephys recordings folder.
    
    tmp = strsplit(fpath, '\');
    upfolder = tmp{end};
    upPath = fpath(1:end-length(upfolder));
    
    i = 1;
    
    % Find all .continuous files within the folder
    AllFile = dir([fpath '\*.continuous']);
    AllFile = struct2cell(AllFile);
    AllFile(2:end, :) = [];
    AllFile = AllFile';

    AllADCch = strfind(AllFile, 'ADC');
    AllADCch = ~cellfun('isempty', AllADCch);
    AllADCch = find(AllADCch ~= 0);

    AllCHch = strfind(AllFile, 'CH');
    AllCHch = ~cellfun('isempty', AllCHch);
    AllCHch = find(AllCHch ~= 0);

    AllAUXch = strfind(AllFile, 'AUX');
    AllAUXch = ~cellfun('isempty', AllAUXch);
    AllAUXch = find(AllAUXch ~= 0);

    if size(AllADCch, 1) == 0
        disp('No trigger recorded');
        trig = 0;
    else
        % Load data for each ADC channel
        for j = 1:size(AllADCch, 1)
            filename = [fpath '\' AllFile{AllADCch(j)}];
            [data_tmp, ~, info] = load_open_ephys_data(filename);
            data(j, :) = data_tmp';
            ADC.label{j} = info.header.channel;
        end
        
        % Remap data based on channel order
        data_remap = zeros(size(data));
        for k = 1:length(ADC.label)
            tmp = ADC.label{k};
            tmp = tmp(4:end);
            data_order(k) = str2num(tmp);
            data_remap(data_order(k), :) = data(k, :);
        end
        data = data_remap;
        trig = data;
    end

    clear data;

    if size(AllCHch, 1) == 0
        disp('No data from amplifier recorded');
        data = 0;
    else
        % Load data for each CH channel
        for j = 1:size(AllCHch, 1)
            filename = [fpath '\' AllFile{AllCHch(j)}];
            [data_tmp, ~, info] = load_open_ephys_data(filename);
            data(j, :) = decimate(data_tmp', 10);
            CH.label{j} = info.header.channel;
        end
        CH.data = data;

        % Remap data based on channel order
        data_remap = zeros(size(data));
        for k = 1:length(CH.label)
            tmp = CH.label{k};
            tmp = tmp(3:end);
            data_order(k) = str2num(tmp);
            data_remap(data_order(k), :) = data(k, :);
        end
        data = data_remap;
    end

    Fs = info.header.sampleRate;

    OrigFolderName = upfolder;
    tmpname = [upPath 'test_' upfolder(end-2:end) '.mat'];
    save(tmpname, 'data', 'trig', 'Fs', 'OrigFolderName', '-v7.3');
    disp([tmpname '... saved'])
end
