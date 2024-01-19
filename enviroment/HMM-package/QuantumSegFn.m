function data = QuantumSegFn(data, PitchVersion,method);
global data;
%% readAudio
% filePathRoot = 'C:\Users\lyk45\OneDrive - Queen Mary, University of London\phd\Projects\Quantum of pitch\dataset\Portamento\';
% AudioPath = strcat(filePathRoot,'Audio\');
% data = readAudioFn(AudioPath);

%% getPitchcurve
% getPitchCurveFn(data)
%% readF0 and get multiple versions （unvoiced pitch indicator: 1 means no; 2 means have）
% data = readMultiF0Fn( filePathRoot,2 );

%% sample pitch curve using different method (Method1 is diff; Method2 is gradient  )
SamplePitchmethodsNum = 1; % SamplePitchmethodsNum: 1(Method1) is diff; 2(Method2) is gradient
data = SamplePitchCurveFn(data, SamplePitchmethodsNum, PitchVersion);

%% plotSamplePoints on one track
plotSamplePoints(data(1),PitchVersion);

%% Quantum information
%-----get the quantum information of time and pitch interval-----
QuantumMode = 0.5; % mode (0.25 is Q; 0.5 is H)
data = getQuantumInfoFn(data,QuantumMode);
%-----get the quantum information of time and pitch interval-----

%-----plot the Frequency and Extent value of each quantum-----
% data(1) = plotFreExtentFn(data(1));
%-----plot the Frequency and Extent value of each quantum-----

%---------------- Get label of segmentations ------------------------------
%%
%-----HMM parameters-----
% ParaHMMobser = struct();
% % we assume transition can be any signal
% ParaHMMobser.Transition.Fre.Limit = [0,data.FsSel/2]; ParaHMMobser.Transition.Fre.LimitConfi = []; ParaHMMobser.Transition.Fre.DistributionType = 'Uniform';
% ParaHMMobser.Transition.Amp.Limit = [0,max(data.quantum.pitchInterval)]; ParaHMMobser.Transition.Amp.LimitConfi = []; ParaHMMobser.Transition.Amp.DistributionType = 'Uniform';
% % we assume vibrato is a group of successive quantum which follow a template, i,e, their frequency, amplitude and
% % intonation are in a limitation.
% ParaHMMobser.vibrato.Rate.Limit = [4,9]; ParaHMMobser.vibrato.Rate.LimitConfi = 0.8; ParaHMMobser.vibrato.Rate.LimitDistributionType = 'Normal';
% ParaHMMobser.vibrato.Extent.Limit = [0.1,2]; ParaHMMobser.vibrato.Extent.LimitConfi = 0.95; ParaHMMobser.vibrato.Extent.LimitDistributionType = 'Normal';
% %we assume steady is a group of successive quantum which follow a template, i,e, their amplitudes are in a limitation.
% ParaHMMobser.Steady.Fre.Limit = [0,data.FsSel]; ParaHMMobser.Steady.Fre.LimitConfi = []; ParaHMMobser.Steady.Fre.LimitDistributionType = 'Uniform';
% ParaHMMobser.Steady.Amp.Limit = [0,0.2]; ParaHMMobser.Steady.Amp.LimitConfi = []; ParaHMMobser.Steady.Amp.LimitDistributionType = 'Normal';
% % we assume there is a minimum of number of quantum or duration of time the
% % group (vibrato or steady) can be perceived
% ParaHMMobser.dur.quantumCriterion = 4; %the frame criterion for the vibrato candidates, make it larger or equal to 6 consecutive frames.
% ParaHMMobser.dur.NotepruningTr = 0.1; % the threshold of the shortest duration of vibrato
% % PDF
% %     % frequency
% %     pdf_F_tran = pdf('Uniform',[0:0.1:],0,data.FsSel/2);
% %     pdf_F_stea = pdf('Uniform',Quantum(i_quan).fre,max(VibratoFreLim),data.FsSel/2);
% %     pdf_F_vibr = pdf('Normal',Quantum(i_quan).fre,sum(VibratoFreLim)./2,5/4)
% %     % amplitude
% %     pdf_A_tran = pdf('Uniform',Quantum(i_quan).pitchInterval,0,max(data.quantum.pitchInterval));
% %     pdf_A_stea = pdf('Uniform',Quantum(i_quan).pitchInterval,0,min(VibratoAmpLim));
% %     pdf_A_vibr = pdf('Normal',Quantum(i_quan).pitchInterval,sum(VibratoAmpLim)./2,3.8/4);
% %     % plot
% %     figure();
% %     plot(0,data.FsSel/2,pdf_F_tran)
% % the name of this method
% Method = 'QuanVibLimIntoHMM';
%----- parameters-----

%%
%-----HMM-----
states = [-1,0,1]'; %the states: Transitory; Steady; Vibration
Nfiles = length(data);
Seg_T = cell(Nfiles,1);
Seg_S = cell(Nfiles,1);
Seg_V = cell(Nfiles,1);


h = waitbar(0,'Vibrato detecting...');
for ifiles = 18
%     ifiles 1:Nfiles
    waitbar(ifiles/Nfiles,h,sprintf('%d%% Vibrato detecting...',round(ifiles/Nfiles*100)));
    Quantum = data(ifiles).quantum;
    %---------Get initial state PDF, transiton matrix and observation matrix
    initialStateDistibution = 1/length(states)*ones(1,length(states));
    obserProb = GetObserProMatrix(data,Quantum,states);
    % transProb = ThreeDGetTransMatrix(Quantum,states);
    transProb = [0.8 0.1 0.1; 0.05 0.9 0.05; 0.005 0.005 0.99 ];
    %------------------------------
    %-----Viterbi----------------
    [path, p] = hmmViterbi_(obserProb, transProb, initialStateDistibution) % use Mochen's code
    %-----Viterbi-----------------

    Seg_T{ifiles} = SegAggreQuantum((path == 1)',Quantum);
    Seg_S{ifiles} = SegAggreQuantum((path == 2)',Quantum);
    Seg_V{ifiles} = SegAggreQuantum((path == 3)',Quantum);
 
end
close(h);
%% Visualization
ifiles = 18;
Segs = [{Seg_T{ifiles}};{Seg_S{ifiles}};{Seg_V{ifiles}}];
segmentsVisualFn(Segs,data(ifiles).F0time,data(ifiles).F0pitch);

%% output segmentations
outputlevel = [1 1 1];
% [data, outputname] = OutputVibratoFn(data,Seg_T,PitchVersion,method,outputlevel);
% [data, outputname] = OutputVibratoFn(data,Seg_S,PitchVersion,method,outputlevel);
[data, outputname] = OutputVibratoFn(data,Seg_V,PitchVersion,method,outputlevel);
