function proportionCorrect = calculateFrameLevelAccuracy(groundTruth, transcribed)
    % Ensure that both arrays are the same length
    if length(groundTruth) ~= length(transcribed)
        error('Ground Truth and Transcribed arrays must be of the same length.');
    end

    % Calculate the number of matching frames
    numMatches = sum(groundTruth == transcribed);

    % Calculate the proportion of correct frames
    proportionCorrect = numMatches / length(groundTruth);
end
