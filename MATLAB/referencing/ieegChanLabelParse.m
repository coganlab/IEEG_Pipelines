function [shankName] = ieegChanLabelParse(inputString)
% Sample input string
%inputString = 'abc123def456ghi789';

% Define a regular expression pattern
pattern = '^(.*?[a-zA-Z]+)(\d+)$';

% Use the regexp function to find the match
match = regexp(inputString, pattern, 'tokens', 'ignorecase');
shankName = [];
if ~isempty(match)
    % Extract the matched portion (excluding the trailing numbers)
    shankName = match{1}{1};
    
    % Display the extracted substring
    fprintf('Matched String: %s\n', shankName);
else
    fprintf('No match found for %s.\n', inputString);
end

end