function [evalMeasures, R]=New_evaluation_mul(listOfGroundTruthFiles,listOfTranscriptionFiles, pitchtrack,outputlevel)
% do modification to allow the function input matrix instead of filename
% Author: Emilio Molina (emm@ic.uma.es)
% In case you use this software tool, please cite the following paper:
% [1] Molina, E., Barbancho A. M., Tardon, L. J., Barbancho, I., "Evaluation
% framework for automatic singing transcription", Proceedings of ISMIR 2014
%
%
% --INPUTS---------------------------------------------
% listOfGroundTruthFiles --> Cell of matrix of the ground-truth.
% See classifyNotes.m to know the accepted formats.
% listOfTranscriptionFiles --> Cell of matrix of the transcription

% Note that two cell arrays must have the same length
%
% --OUTPUTS:---------------------------------------------------------------
% evalMeasures contains all relevant information described in [1], averaged
% for the whole dataset.


for i_file=1:length(listOfGroundTruthFiles)
    %     i_file
    fileTranscription=listOfTranscriptionFiles{i_file};
    fileGroundTruth=listOfGroundTruthFiles{i_file};
    filepitchtrack = pitchtrack{i_file};
%     R = struct();
    switch outputlevel
        case 'Framelevel'
            if any(fileTranscription==1) && any(fileGroundTruth==1)
                r = classifyFrame_mul(fileTranscription,fileGroundTruth, filepitchtrack)
                R(i_file) = r;
            end
        case 'Notelevel'
            if ~isempty(fileTranscription)==1&&~isempty(fileGroundTruth)==1 % avoid empty
                r =classifyNotes_mul(fileTranscription,fileGroundTruth, filepitchtrack)
                R(i_file) = r;
                %         disp(sprintf('Processing... %i / %i',i,length(listOfGroundTruthFiles)));
            end
    end
end
R
% [R(1:end).N_GT]'
evalMeasures = SumEvaResults(R,outputlevel);




end