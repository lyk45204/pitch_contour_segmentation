%% this function evaluation_all is to evaluate multiple methods at the same time
% Input: listOfTranscriptionFiles (can have more than one methods)
% output:
function [Metrics, Results] = New_evaluation_all(data, listOfGroundTruthFiles, listOfTranscriptionFiles, pitchtrack, outputlevel)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% initialization
Nmethods = size(listOfTranscriptionFiles,2);
Ncolumn_Metrics = Nmethods + 1; 

switch outputlevel
    case 'Framelevel'
        load('Metrics_Vibrato_rowname_frame.mat');
        Metrics_rowname = Metrics_Vibrato_rowname_frame;
    case 'Notelevel'
        load('Metrics_Vibrato_rowname_note.mat');
        Metrics_rowname = Metrics_Vibrato_rowname_note;
end

Nrow_Metrics = length(Metrics_rowname);

Metrics = cell(Nrow_Metrics+1,Ncolumn_Metrics);
Metrics(2:end,1) = Metrics_rowname;
Metrics_methodname = fieldnames(data(1).VibratoDetResults);
Metrics(1,2:end) = Metrics_methodname';
R = cell(Nmethods,1);
% evaluation
for i_methods = 1:Nmethods
    [evalMeasures, R{i_methods}] = New_evaluation_mul(listOfGroundTruthFiles, listOfTranscriptionFiles(:,i_methods), pitchtrack, outputlevel);
    Metrics(2:end,i_methods+1) = num2cell(evalMeasures);
    
end

Results = R;