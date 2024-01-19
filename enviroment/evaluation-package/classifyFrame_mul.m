function Results = classifyFrame_mul(fileTranscription,fileGroundTruth, filepitchtrack)
% ----------------------------------------------------------------------
%
% Results = classifyFrame(fileTranscription,fileGroundTruth) return a set
% of evaluation measures (within a struct variable) that represents the
% transcription performance of fileTranscription (MIDI or ASCII-formatted
% file) with respect to fileGroundTruth.
% 
% -- INPUTS --------------------------------
% Both fileTranscription and fileGroundTruth are monophonic melodies.The 
% format is: 0 indicates no vibrato in this frame while 1 means yes
%
%         0000001111110000000
%

%
% -- OUTPUT -------------------------------
% The ouput Results is a struct 
% Results.Precision --> true positive vibrato frames / positive vibrato frames (Est vibrato frames)
% Results.Recall --> true positive vibrato frames / total vibrato frames (GT vibrato frames)
% Results.Fmeasure --> 2*P*R/(P+R)

% true positive vibrato frames
sumVectors = fileTranscription+fileGroundTruth;
Ture_idx_vector = find(sumVectors==2);
NumTure = length(Ture_idx_vector);
% positive vibrato frames
Positve_idx_vector = find(fileTranscription==1);
NumPositve = length(Positve_idx_vector);
% total vibrato frames
Total_idx_vector = find(fileGroundTruth==1);
NumTotal = length(Total_idx_vector);

% ---- Write output struct Results:
Results.Precision = NumTure/NumPositve;
Results.Recall = NumTure/NumTotal;
if Results.Precision==0 && Results.Recall==0
    Results.Fmeasure = 0;
else
    Results.Fmeasure = 2*Results.Precision*Results.Recall/(Results.Precision+Results.Recall);
end










