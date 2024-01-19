% HMM-based model parameters optimization using k-fold cross validation
% estimate parameters from my annotation labels on Jingju vocal pitch tracks.

clc; clear all; close all;
set(groot, 'defaultAxesFontSize', 18);
set(groot, 'defaultTextFontSize', 18);
set(groot, 'DefaultLineLineWidth', 2); % Sets the default line width to 2

% set parameters by observation from the quantum distribution
%% readAudio
filePathRoot = ['/Users/yukunli/Desktop/PhD/Projects/Quantum of pitch/Dataset/SMC2016_noteseg/SMC2016-master/samllDataset/Train/'];
AudioPath = strcat(filePathRoot,'Audio/');

global data;
DatasetName = 'Jingju SMC 2016 Small Version Training Set';
data = readFileNameFn(AudioPath);
% data = readAudioFn(AudioPath);

%% read segments of steady, vibrato, transitory and noises annotations
BasicSegPath = strcat(filePathRoot,'Seg_My/BasicSeg/');
data = readSegFn(BasicSegPath,data);

% %% readF0
% data = readF0Fn(filePathRoot,1);


%% readF0 and get multiple versions （unvoiced pitch indicator: 1 means no; 2 means have）
data = readMultiF0Fn( filePathRoot,1 );

%% select a version of pitch trace
pitchversion = 'InterpoSmoothPitch';
data = PitchVersionSelectFn(data,pitchversion);



%% extract features
stateName = {'transitory', 'steady', 'modulation', 'noise'};
labels = {-1, 0, 1, -2}
[features, quanta] = extract_quantums_and_features(data, stateName, labels); % include nosie state segment

%% feautures distribution analysis
% [resolutions,bandwidths_default,cov_matrixs] = FeaturesAnalysis(stateName, features); 
close all;

%% Extract Pitch Contours including quantums with labels(no 'noise' and remove the silence region) 
pitchContourSegments = ExtractPitchContours(data,quanta);


%% Estimate the best HMM Parameters
stateName = {'transitory', 'steady', 'modulation', 'noise'};
statelabels = {-1, 0, 1, -2};
k = 10; % k-fold validation
bandwidthMultipliers = [1:10]; 
% scaling
repeatTimes = 3;
[BestParams,BestPerformance] = optimizeHMMParameters(stateName, statelabels, pitchContourSegments, k, ...
    bandwidthMultipliers, repeatTimes);

 

%%%%%%%%%%%%%%
%% GMM fitting (a possible alternative option for observation probability estimation)
% maxComponents = 10; % set here but will find the best in the function
% [observation_likelihoods_Distr] = estimate_observation_likelihoods(features,stateName,maxComponents);
% 
% binWidthFre = 1;
% binWidthExtent = 0.05;
% plotFeaturesWithGMM3D(features, states, observation_likelihoods_Distr,binWidthFre,binWidthExtent);

function [transitionMatrix, observationLikelihoods, bandwidth] = trainHMM_KDE(trainingSet, stateName, statelabels, i_band_Multiplier)


transitionMatrix = [];
observationLikelihoods = struct();

%% Estimate transition probabilities
[filteredQuantaByState, quantaList] = extractQuanta(trainingSet)

transitionMatrix = estimate_transition_matrix(quantaList, stateName, statelabels);

%% Estimate observation probabilities distributions
features = extractFeatures(filteredQuantaByState)
[observationLikelihoods, bandwidth] = estimate_observation_distribution(features, stateName, i_band_Multiplier);
 

end

function [observationLikelihoods, bandwidths] = estimate_observation_distribution(features, stateName, i_band_Multiplier)
    num_states = length(stateName);
    observationLikelihoods = struct(); % 初始化 observationLikelihoods 结构体
    bandwidths = [];

    for i = 1:num_states
        state = stateName{i};
        if ~strcmp(state, 'noise') % 检查状态是否不是噪声
            state_features = features.(state); % 获取特定状态的特征
         
            % 使用 kde2d 函数计算密度分布
            [bandwidth, density, X, Y] = kde2d(state_features, i_band_Multiplier);

            % 存储密度分布及相关参数
            observationLikelihoods.(state).density = density;
            observationLikelihoods.(state).X = X;
            observationLikelihoods.(state).Y = Y;
            bandwidths = [bandwidths;bandwidth];
        end
    end
end





function features = extractFeatures(filteredQuantaByState)
    % 获取 filteredQuantaByState 中的所有状态名
    stateNames = fieldnames(filteredQuantaByState);
    features = struct(); % 初始化 features 结构体

    % 遍历每个状态
    for i = 1:length(stateNames)
        currentState = stateNames{i};
        currentStateData = filteredQuantaByState.(currentState);

        % 检查当前状态下是否有数据
        if ~isempty(currentStateData)
            % 预分配矩阵以存储 'dur' 和 'extent' 的值
            currentStateFeatures = zeros(numel(currentStateData), 2);

            % 遍历当前状态下的每个元素
            for j = 1:numel(currentStateData)
                % 提取 'dur' 和 'extent' 的值
                currentStateFeatures(j, 1) = currentStateData(j).dur;
                currentStateFeatures(j, 2) = currentStateData(j).extent;
            end

            % 将提取的特征存储在 features 结构体中
            features.(currentState) = currentStateFeatures;
        else
            % 如果当前状态下没有数据，分配空矩阵
            features.(currentState) = [];
        end
    end
end


function [filteredQuantaByState, quantaList] = extractQuanta(dataset)
    % 初始化一个空结构体数组
    quantaList = struct();

    % 定义一个索引，用于追踪在 quantaList 中添加的元素数量
    index = 1;

    for i = 1:numel(dataset)
        % 对于每个 Quantum 结构体数组的元素
        for j = 1:numel(dataset(i).Quantum)
            % 获取 Quantum 中的所有字段名
            fieldNames = fieldnames(dataset(i).Quantum(j));
            % 为每个字段名创建一个字段，并将对应的值赋给 quantaList
            for k = 1:numel(fieldNames)
                fieldName = fieldNames{k};
                quantaList(index).(fieldName) = dataset(i).Quantum(j).(fieldName);
            end
            % 更新 index，以便于下一个 Quantum 元素的添加
            index = index + 1;
        end
    end

    % 从 quantaList 中提取基于状态的 filteredQuantaByState
    stateName = {'transitory', 'steady', 'modulation', 'noise'}; % 状态名列表
    filteredQuantaByState = struct(); % 初始化一个空结构体用于存储基于状态过滤的 quanta

    % 根据每个状态名循环
    for i = 1:length(stateName)
        currentState = stateName{i};
        % 根据当前状态过滤 filteredQuantaByState
        filteredQuantaByState.(currentState) = quantaList(strcmp({quantaList.state}, currentState));
    end
end





function pitchContourSegments = ExtractPitchContours(data,quanta)
    % Initialize the output structure
    pitchContourSegments = struct('F0time', {}, 'F0pitch', {}, 'Quantum', {}, 'QuantumNum', {}, 'Filename', {});

    % Process each file in the data
    for fileIdx = 1:length(data)
        fileIdx
        F0originalTime = data(fileIdx).OriPitch.time;

        F0time = data(fileIdx).F0time;
        F0pitch = data(fileIdx).F0pitch;
        % Find indices of quanta that belong to the specified file
        indices = find([quanta.file] == fileIdx);
        % Extract those quanta of the current file
        quantumData = quanta(indices);
        filename = data(fileIdx).Filename;

        % Find breaks in F0time to identify pitch contour segments
        delta_time = F0originalTime(2)-F0originalTime(1);
        breaks_Ori = find(diff(F0originalTime) > 1.2*delta_time);
        breaks_Time_end = F0originalTime(breaks_Ori);
        breaks_Time_start = F0originalTime(breaks_Ori+1);
        % set noises region as breaks
        noiseIndices = find(arrayfun(@(x) strcmp(x.state, 'noise'), quantumData));
        if ~isempty(noiseIndices)
            noise_quantum_onset = quantumData(noiseIndices).onset;
            noise_quantum_offset = quantumData(noiseIndices).offset;
            breaks_Time_end = [breaks_Time_end;noise_quantum_onset];
            breaks_Time_start = [breaks_Time_start;noise_quantum_offset];
        end



        breaks_loc_start = zeros(length(breaks_Time_start),1);
        breaks_loc_end = zeros(length(breaks_Time_end),1);

        for i_breaks = 1:length(breaks_loc_start)
            [v_min,breaks_loc_start(i_breaks)] = min( abs(breaks_Time_start(i_breaks)-F0time) );
            [v_min,breaks_loc_end(i_breaks)] = min( abs(breaks_Time_end(i_breaks)-F0time) );

        end

        segmentStarts = sort([1; breaks_loc_start]);
        segmentEnds = sort([breaks_loc_end; length(F0time)]);

        % Create segments
        for i_segment = 1:length(segmentStarts)
            i_segment
            startIdx = segmentStarts(i_segment); % ind of F0
            endIdx = segmentEnds(i_segment);
            
            % Initialize segment
            segment = struct();
            segment.F0time = F0time(startIdx:endIdx);
            segment.F0pitch = F0pitch(startIdx:endIdx);
            segment.Quantum = struct();
            segment.Filename = filename;

            % Find quanta within this segment's time range
            segmentStartTime = F0time(startIdx);
            segmentEndTime = F0time(endIdx);

            if ~isempty(quantumData)
                % Initialize segment.Quantum with the same fields as the first quantumData
                segment.Quantum = quantumData(1);
                segment.Quantum(1) = []; % Clear the initial content
            else
                error('no quantum in the segment %d in file %d',i_segment,fileIdx)
            end
            
            for j = 1:length(quantumData)
                quantumOnset = quantumData(j).onset;
                quantumOffset = quantumData(j).offset;

                if ~strcmp(quantumData(j).state,'noise' ) && ...
                        ((quantumOnset >= segmentStartTime && quantumOnset <= segmentEndTime) || ...
                        (quantumOffset >= segmentStartTime && quantumOffset <= segmentEndTime))
                    segment.Quantum(end+1) = quantumData(j);
                    
                end
            end
            segment.QuantumNum = length(segment.Quantum);

            % update the boundary of F0time and F0pitch of the pitch
            % contour to remove the silence region.
            if segment.QuantumNum>0
                quanta_in_seg = segment.Quantum;
                onset = quanta_in_seg(1).onset;
                offset = quanta_in_seg(end).offset;

                F0time_seg = segment.F0time;
                F0pitch_seg = segment.F0pitch;

                if F0time_seg(1)<onset
                    [~, onsetIdx] = min(abs(F0time_seg - onset));
                    F0time_seg = F0time_seg(onsetIdx:end);
                    F0pitch_seg = F0pitch_seg(onsetIdx:end);
                end
                if F0time_seg(end)>offset
                    [~, offsetIdx] = min(abs(F0time_seg - offset));
                    F0time_seg = F0time_seg(1:offsetIdx);
                    F0pitch_seg = F0pitch_seg(1:offsetIdx);
                end

                segment.F0time = F0time_seg;
                segment.F0pitch = F0pitch_seg;
            end

            % Append to output structure
            if segment.QuantumNum > 0 % ignore the silent pitch contour segment without annotation
                pitchContourSegments(end+1) = segment;
            end

        end
    end

   

    % Check if any pitchContourSegments share the same quantum region
    % If yes, display an error and show the numbers of the pitchContours and the shared state

    numSegments = length(pitchContourSegments);

    for i = 1:numSegments-1
%         i
            % Get the state regions for the current pair of segments
            Quantum_i = pitchContourSegments(i).Quantum;
            Quantum_j = pitchContourSegments(i+1).Quantum;

            % Check for shared states
            if Quantum_i(end).onset == Quantum_j(1).onset
                error('Shared Quantum found between segments %d and %d in file %d', i, i+1,fileIdx);
            end

    end

    
end



function [bestSegmentPieces, minVariance] = partitionPitchContours(pitchContourSegments, k)

partition_times = 100;
bestVariance = Inf;
bestSegmentPieces = {};

totalSegments = length(pitchContourSegments);

for i_partition = 1:partition_times
    % Generate random indices
    randomIndices = randperm(totalSegments);
    segmentPieces = cell(1, k);

    % Partition data
    for i = 1:k
        startIndex = round((i-1) * totalSegments / k) + 1;
        endIndex = round(i * totalSegments / k);
        segmentIndices = randomIndices(startIndex:endIndex);
        segmentPieces{i} = pitchContourSegments(segmentIndices);
    end

    % Calculate variance for this partition
    variance = calculateQuantumVariance(segmentPieces);

    % Update bestSegmentPieces if current partition has smaller variance
    if variance < bestVariance
        bestVariance = variance;
        bestSegmentPieces = segmentPieces;
    end
end

% Return the best set of segment pieces and its variance
minVariance = bestVariance;

end

% Function to calculate quantum number variance
function variance = calculateQuantumVariance(segmentPieces)
    quantumNumbers = zeros(1, length(segmentPieces));
    for i = 1:length(segmentPieces)
        quantumNumbers(i) = sum(arrayfun(@(x) x.QuantumNum, segmentPieces{i}));
    end
    variance = var(quantumNumbers);
end





function [BestParams,BestPerformance] = optimizeHMMParameters(stateName, statelabels, data, k, bandwidthMultipliers,repeatTimes)

[bestSegmentPieces, minVariance] = partitionPitchContours(data, k);
segmentPieces = bestSegmentPieces;

% Initialize the best parameters and their evaluation metrics
BestParams = struct('bandwidthMultiplier', [], ...
    'bandwidth', [],'transitionMatrix',[],'observationLikelihoods',[]);
BestPerformance = -inf;

Performance = [];
for i_band_Multiplier = bandwidthMultipliers

    avgPerformance = zeros(repeatTimes,1);
    varPerformance = zeros(repeatTimes,1);

    for i_repeat = 1:repeatTimes
        % k-fold cross-validation
        performance_frame_mean_fold = zeros(k,1);

        for i = 1:k

            % Start the timer
            tic;

            % Display progress
            progress = ( ((i_band_Multiplier-1)*repeatTimes*k + (i_repeat-1)*k + i) / ...
                (max(bandwidthMultipliers)*repeatTimes*k) ) * 100;
            fprintf('Progress: %.2f%%\n', progress);

            % Split data into training and validation sets
            validationSet = segmentPieces{i};
            trainingSet = [segmentPieces{1:i-1} segmentPieces{i+1:end}];

            % Train HMM with the current parameters
            [transitionMatrix, observationLikelihoods, bandwidth] = trainHMM_KDE(trainingSet, stateName, statelabels, i_band_Multiplier);

            % Validate the HMM model
            performance_frame_mean_fold(i) = validateHMM(transitionMatrix, observationLikelihoods, validationSet);
            
            % Stop the timer and display the elapsed time
            elapsedTime = toc;
            fprintf('Elapsed time: %.2f seconds\n', elapsedTime);

        end
        avgPerformance(i_repeat) = mean(performance_frame_mean_fold);
        varPerformance(i_repeat) = var(performance_frame_mean_fold);
    end

    para_perfor_mean = mean(avgPerformance);
    para_perfor_var = mean(varPerformance);
    lamda = 1/.10;
    para_score_performance = para_perfor_mean - lamda * para_perfor_var;

    if para_score_performance > BestPerformance
        BestPerformance = para_score_performance;
        BestParams.bandwidthMultiplier = i_band_Multiplier;
        BestParams.bandwidth = bandwidth;
        BestParams.transitionMatrix = transitionMatrix;
        BestParams.observationLikelihoods = observationLikelihoods;
    end

    
end

end


function performance_frame_mean = validateHMM(transitionMatrix, observationLikelihoods, validationSet)

num_contour = numel(validationSet);
proportionCorrect = zeros(num_contour,1);
for i_contour = 1:num_contour
    % cut quantum
    contour_seg = validationSet(i_contour);
    F0time_seg = contour_seg.F0time;
    F0pitch_seg = contour_seg.F0pitch;

    seg_pitchtrack = [F0time_seg F0pitch_seg];

    quanta = struct('file', {}, 'state', {}, 'pitchtrack', {}, 'onset', {}, 'offset', {}, ...
        'dur', {}, 'pitchinterval', {}, 'fre', {}, 'extent', {});
    MinPeakDistance = 0;
    [peaks troughs extremums] = FindLocalextremumsFn(seg_pitchtrack(:,2),seg_pitchtrack(:,1),MinPeakDistance,1);
    extremums_time = extremums(:,1);
    num_quantums = size(extremums,1)-1;
    QuantumSeg_matrix = [];

    observation_seq = [];
    for i_quantum = 1:num_quantums
        % Extract features and details for each quantum
        quantum_onset = extremums_time(i_quantum);
        quantum_offset = extremums_time(i_quantum+1);
        quantum_dur = quantum_offset - quantum_onset;
        quantum_pitchtrack = cell2mat(cutpitch(quantum_onset, quantum_offset, seg_pitchtrack));

        dur = quantum_dur;
        pitchinterval = (quantum_pitchtrack(end,2)-quantum_pitchtrack(1,2));
        fre = 1./(quantum_dur*2);
        extent = pitchinterval./2;

        % Assign quantum details in a more concise manner
        quantum = struct('file', contour_seg.Filename, 'state', 0, ...
            'pitchtrack', quantum_pitchtrack, 'onset', quantum_onset, 'offset', quantum_offset, ...
            'dur', dur, 'pitchinterval',pitchinterval,'fre', fre, 'extent', extent);
        quanta(end+1) = quantum;
        QuantumSeg_matrix = [QuantumSeg_matrix; quantum.onset quantum.offset];

        % get the observation sequence of this pitch contour
        observation_seq = [observation_seq; dur fre];
    end

    validationSet(i_contour).QauntumEstimate = quanta;


    % HMM segmentation
    hiddenStates = {'transitory', 'steady', 'modulation'};
    hiddenStatesLabel = {-1, 0, 1};
    hiddenLabel2State = containers.Map(hiddenStatesLabel, hiddenStates);
    hiddenState2Label = containers.Map(hiddenStates, hiddenStatesLabel);


    numStates = numel(hiddenStates);
    initialProb = repmat(1/numStates,numStates,1);
    bestSequence = HMM(hiddenStates, hiddenStatesLabel, observation_seq, initialProb, transitionMatrix, observationLikelihoods);
    bestSequence = bestSequence-2;
    for i_quantum = 1:num_quantums
        validationSet(i_contour).QauntumEstimate(i_quantum).state = hiddenLabel2State(bestSequence(i_quantum)); 
    end
    
    % merge quantum-level segment sequence to state-level 
    QuantumSeg_matrix = [QuantumSeg_matrix bestSequence'];
    TrStateSeg_matrix = mergeQuantumSegments(QuantumSeg_matrix);
    validationSet(i_contour).TrStateSeg = TrStateSeg_matrix;

    % convert segment-level into frame-level
    ContourTimeStamp = validationSet(i_contour).F0time;
    validationSet(i_contour).TrFrame = convertToFrameLevel(TrStateSeg_matrix, ContourTimeStamp);

    % for ground truth, merge quantum-level segment sequence to state-level 
    GTquanta = validationSet(i_contour).Quantum;
    num_quanta = numel(GTquanta);
    GTQuantumSeg_matrix = zeros(num_quanta,3);
    for i_quanta = 1:num_quanta
        GTQuantumSeg_matrix(i_quanta,1) = GTquanta(i_quanta).onset;
        GTQuantumSeg_matrix(i_quanta,2) = GTquanta(i_quanta).offset;
        GTQuantumSeg_matrix(i_quanta,3) = hiddenState2Label(GTquanta(i_quanta).state);
    end
    
    GTStateSeg_matrix = mergeQuantumSegments(GTQuantumSeg_matrix);
    validationSet(i_contour).GTStateSeg = GTStateSeg_matrix;

    % convert segment-level into frame-level
    ContourTimeStamp = validationSet(i_contour).F0time;
    validationSet(i_contour).GTFrame = convertToFrameLevel(GTStateSeg_matrix, ContourTimeStamp);


    % performance of detection method
    % frame_level metrics
    groundTruth_frame = validationSet(i_contour).GTFrame(:,2);
    transcribed_frame = validationSet(i_contour).TrFrame(:,2);
    proportionCorrect(i_contour) = calculateFrameLevelAccuracy(groundTruth_frame, transcribed_frame);
        
end

performance_frame_mean = mean(proportionCorrect);

end

function Frame_matrix = convertToFrameLevel(StateSeg_matrix, ContourTimeStamp)
    % Initialize Frame_matrix
    Frame_matrix = zeros(length(ContourTimeStamp), 2);
    
    % Assign time values to the first column of Frame_matrix
    Frame_matrix(:, 1) = ContourTimeStamp;
    
    % Iterate through each timestamp
    for i = 1:length(ContourTimeStamp)
        currentTime = ContourTimeStamp(i);

        % Find the segment that the current time belongs to
        for j = 1:size(StateSeg_matrix, 1)
            if currentTime >= StateSeg_matrix(j, 1) && currentTime <= StateSeg_matrix(j, 2)
                Frame_matrix(i, 2) = StateSeg_matrix(j, 3); % Assign statelabel
                break; % Exit the loop once the segment is found
            end
        end
    end
end


function StateSeg_matrix = mergeQuantumSegments(QuantumSeg_matrix)
    % Initialize the output matrix with the first row of the input matrix
    StateSeg_matrix = QuantumSeg_matrix(1,:);

    % Iterate through the QuantumSeg_matrix
    for i = 2:size(QuantumSeg_matrix, 1)
        currentSeg = QuantumSeg_matrix(i,:);
        lastSeg = StateSeg_matrix(end,:);

        % Check if the current segment can be merged with the last segment
        if currentSeg(3) == lastSeg(3)
            % Merge segments: Update the offset of the last segment
            StateSeg_matrix(end, 2) = currentSeg(2);
        else
            % Different state label, so add the current segment as a new row
            StateSeg_matrix = [StateSeg_matrix; currentSeg];
        end
    end
end





function [X_scaled,med,iqrValue] = robustScaling(X)
    % Calculate the median of the dataset
    med = median(X);

    % Calculate the interquartile range of the dataset
    iqrValue = iqr(X);

    % Perform the robust scaling
    X_scaled = (X - med) ./ iqrValue;

    % Ensure element-wise operations for each feature
end

function [cond_transitory,cond_steady,cond_modulation] = calculate_condition_numbers(cov_matrixs)
    % Define the covariance matrices
    transitory_cov = cov_matrixs{1};
    steady_cov = cov_matrixs{2};
    modulation_cov = cov_matrixs{3};
    
    % Calculate the condition numbers
    cond_transitory = cond(transitory_cov);
    cond_steady = cond(steady_cov);
    cond_modulation = cond(modulation_cov);
    
    % Display the results
    disp(['Condition number for transitory matrix: ', num2str(cond_transitory)]);
    disp(['Condition number for steady matrix: ', num2str(cond_steady)]);
    disp(['Condition number for modulation matrix: ', num2str(cond_modulation)]);
end
function [resolutions,bandwidths,cov_matrixs]= FeaturesAnalysis(stateName, features, params)
    
    resolutions = [];
    bandwidths = [];

    kernelType = {'normal','normal','normal','normal'};

    num_states = length(stateName);
    features_scaled = struct();

    
    cov_matrixs = cell(num_states, 1); % 初始化用于存储协方差矩阵的单元数组

    for i = 1:num_states
        state = stateName{i};
        if ~strcmp(state, 'noise') % 检查状态是否不是噪声
            state_features = features.(state); % 获取特定状态的特征
            
            % 1D
            figure;
            params = {}; 
            % Histogram for Duration
            subplot(2, 1, 1);
            h1 = histogram(state_features(:, 1),'Normalization','pdf');
            hold on;
%             ksdensity(state_features(:, 1),'Kernel', kernelType{i});
            hold off;
            edges1 = h1.BinEdges;
            title(['Histogram of duration in ', state, ' state']);
            xlabel('Duration/s');
            ylabel('pdf');


            % Histogram for Pitch Interval/Semitone
            subplot(2, 1, 2);
            h2 = histogram(state_features(:, 2),'Normalization','pdf');
            hold on;
%             ksdensity(state_features(:, 2),'Kernel', kernelType{i});
            hold off;
            edges2 = h2.BinEdges;
            title(['Histogram of extent in ', state, ' state']);
            xlabel('Extent/Semitone');
            ylabel('pdf');

            saveas(gcf, ['Histogram of duration and extent in ', state, ' state', '.png']);


            %2D
            figure;
            % 特征在不同状态下的分布
            resolution1 = abs(edges1(2)-edges1(1));
            resolution2 = abs(edges2(2)-edges2(1));

            %params = { 'PlotFcn', 'contour', 'Support', [0 -Inf; Inf Inf], 'BoundaryCorrection', 'reflection'};
            params = { 'PlotFcn', 'contour'};    
            [f, xi, bw, Y] = ksdensity(state_features,'Kernel', kernelType{i},params{:}); % MATLAB automatically selects grid points
            
%             hold on;
            scatter(state_features(:, 1), state_features(:, 2)); % 绘制散点图
%             hold off;
            xlabel('Duration/s'); % 设置X轴标签
            ylabel('Extent/Semitone'); % 设置Y轴标签
            title(['Scatter plot of feature distribution in ', state, ' state']); % 设置描述性标题
            saveas(gcf, ['Scatter Plot of fatures in ', state, ' state', '.png']);
            % and estimated Contour Plot

            resolutions = [resolutions;resolution1,resolution2];
            bandwidths = [bandwidths;bw];

            % 3D Histogram Plot for both features
            figure;
            subplot(1,2,1)
            % Calculate the total number of data points
            totalDataPoints = size(state_features, 1);
            


            % Use hist3 to get histogram counts and then normalize to get probabilities
            [N, C] = hist3(state_features, 'Edges', {edges1, edges2}); % Adjust the bin sizes as needed
            dx = edges1(2) - edges1(1);
            dy = edges2(2) - edges2(1);
            binArea = dx * dy;
            N = N / totalDataPoints / binArea; % Normalize to get probabilities

            % Plot using bar3 to visualize probabilities
            h = bar3(N');
            xlabel('Duration/s');
            ylabel('Extent/Semitone');
            zlabel('pdf');
            title(['3D Histogram and estimated pdf in ', state, ' state']);
            
            % Adjust X and Y tick labels to match bin edges
            set(gca, 'XTick', 1:numel(edges1), 'XTickLabel', arrayfun(@num2str, edges1, 'UniformOutput', false));
            set(gca, 'YTick', 1:numel(edges2), 'YTickLabel', arrayfun(@num2str, edges2, 'UniformOutput', false));

            % Adjust the color of each bar based on its height for better contrast
            for k = 1:length(h)
                zdata = h(k).ZData;
                h(k).CData = zdata;
                h(k).FaceColor = 'interp';
            end
            subplot(1,2,2)
            hold on;
            [x1,x2] = meshgrid(edges1, edges2);
            x1 = x1(:);
            x2 = x2(:);
            gridxi = [x1 x2];
            params = {xi};
            
            ksdensity(state_features,gridxi, 'Kernel', kernelType{i});
            hold off;

             

            % 计算feature之间的协方差
            cov_matrix = cov(state_features(:, 1), state_features(:, 2));

            % 存储协方差矩阵
            cov_matrixs{i} = cov_matrix;

            % normalization using robustScaling
%             state_features_scaled = robustScaling(state_features);
%             features_scaled.(state) = state_features_scaled;
% 
%             cov_matrix_scaled = cov(state_features(:, 1), state_features(:, 2));
% 
%             % 存储协方差矩阵
%             cov_matrixs_scaled{i} = cov_matrix_scaled;
        end
    end
end




function Ranges = calculateFeatureStats(stateFeatures,states)

Ranges = zeros(3,2);
% Iterating over each field
for i = 1:length(states)
    state = states{i};
    fprintf('State: %s\n', state);

    % Accessing the data for each state
    data = stateFeatures.(state);
    
    % Calculating range, mean, and standard deviation for each column (feature)
    for col = 1:size(data, 2)
        min_value = min(data(:, col));
        max_value = max(data(:, col));
        range_value = range(data(:, col));
        mean_value = mean(data(:, col));
        std_deviation = std(data(:, col));

        Ranges(i,col) = range_value;

        % Displaying the results for each feature
        fprintf('Feature %d - Min: %f, Max: %f, Range: %f, Mean: %f, Standard Deviation: %f\n', col, min_value, max_value, range_value, mean_value, std_deviation);
    end

    fprintf('\n'); % New line for better readability
end
end

function plotFeaturesWithGMM3D(stateFeatures, stateName, observation_likelihoods_Distr,binWidthX, binWidthY)
    % 遍历所有状态
    Ranges = calculateFeatureStats(stateFeatures,stateName);
    for i_state = 1:3
        state_features = stateFeatures.(stateName{i_state});
        % 获取 GMM 模型
        gmModel = observation_likelihoods_Distr{i_state};

        % 创建并显示三维直方图

        % 指定直方图的分箱数量
        RangeX = Ranges(i_state,1);
        RangeY = Ranges(i_state,2);

        numBinsX = ceil(RangeX / binWidthX);
        numBinsY = ceil(RangeY / binWidthY);
        % 使用 hist3 获取频数直方图
        [counts, centres] = hist3(state_features, [numBinsX, numBinsY]);

        
        % 将计数转换为概率
        % 计算每个箱子的面积
        dx = centres{1}(2) - centres{1}(1);
        dy = centres{2}(2) - centres{2}(1);
        binArea = dx * dy;

        % 转换为概率密度
        probDensity = counts / sum(counts(:)) / binArea;

        % 使用 mesh 或 surf 绘制概率密度

        [xGrid, yGrid] = meshgrid(centres{1}, centres{2});
    

        figure;
        surf(xGrid, yGrid, probDensity', 'EdgeColor', 'red');
        view(3);
        xlabel('Feature 1');
        ylabel('Feature 2');
        zlabel('Probability Density');
        title(['3D Histogram with GMM PDF Overlay for State: ' stateName{i_state}]);
        colormap jet;
        % 使用正交投影
        camproj('perspective');
        alpha(0.8); % 设置透明度
        

%         % 设置直方图的视图和颜色
%         set(get(gca,'child'),'FaceColor','interp','CDataMode','auto');
% 
%         %侧面视图
%         view(3);
% 
%         % 设置透视效果
%        

        % 创建网格以绘制 GMM PDF
        figure;
        gridPoints = [xGrid(:), yGrid(:)];

        % 计算网格上每点的 GMM PDF
        pdfValues = reshape(pdf(gmModel, gridPoints), size(xGrid));

        % 叠加 GMM PDF
        surf(xGrid, yGrid, pdfValues, 'EdgeColor', 'black'); % 使用 mesh 而不是 contour

        % 设置图表的属性
        xlabel('Feature 1');
        ylabel('Feature 2');
        zlabel('Probability Density');
        title(['3D Histogram with GMM PDF Overlay for State: ' stateName{i_state}]);
        colormap jet;
        % 使用正交投影
        camproj('perspective');
        alpha(0.8); % 设置透明度
        

%         figure;
%         X = state_features(:,1);
%         Y = state_features(:,2);
%         hold on;
%         ezcontour(@(x,y)pdf(gmModel,[x y]), [min(X) max(X)],[min(Y) max(Y)]);
%         scatter(X,Y,10,'ko'); % 原始数据点
%         hold off;
    end
end

function [bestGMM, bestNumComponents, scores] = kFoldValidationWithPlot(state_features, maxComponents, k)
    % 初始化分数存储数组
    scores = zeros(maxComponents, 1);
    
    % 遍历不同的组件数量
    for numComponents = 1:maxComponents
        cvScores = zeros(k, 1);
        
        % 生成 k-fold 交叉验证的索引
        indices = crossvalind('Kfold', size(state_features, 1), k);
        
        % k-fold 交叉验证
        for i = 1:k
            testIdx = (indices == i); 
            trainIdx = ~testIdx;
            testData = state_features(testIdx, :);
            trainData = state_features(trainIdx, :);
            
            % 训练 GMM 模型
            gmmModel = fitgmdist(trainData, numComponents, 'RegularizationValue', 0.1);
            
            % 在测试集上评估模型
            cvScores(i) = -sum(log(pdf(gmmModel, testData))); % 对数似然
        end
        
        % 计算平均分数
        scores(numComponents) = mean(cvScores);
    end
    
    % 找到最佳分数和组件数量
    [bestScore, bestNumComponents] = min(scores);
    bestGMM = fitgmdist(state_features, bestNumComponents, 'RegularizationValue', 0.1);
    
    % 绘制图形
    figure;
    plot(1:maxComponents, scores, '-o');
    xlabel('Number of Components');
    ylabel('Mean Log Likelihood Score');
    title('K-Fold Validation Scores for GMM Components');
    grid on;
end



%%%%%%%%%%%% feature extraction %%%%%%%%%%%%%%%%%%%

function [features, quanta] = extract_quantums_and_features(data, stateNames, labels)

transitory_cutoff = 1;%Hz 
steady_cutoff = 200; %Hz
vibrato_cutoff = 20; % Hz

    features = struct();
    quanta = struct('file', {}, 'state', {}, 'pitchtrack', {}, ...
        'onset', {}, 'offset', {}, 'dur', {}, 'pitchinterval', {}, 'fre', {}, 'extent', {});
    label_to_state = containers.Map(labels, stateNames);

    for i_files = 1:length(data)
        annotated_segments = data(i_files).State;
        file_pitchtrack = [data(i_files).F0time data(i_files).F0pitch];
        label_vector = annotated_segments(:,2); % -1, 0, 1, -2
        quantum_index = 1;

        for i_seg = 1:size(annotated_segments, 1)

            seg_onset = annotated_segments(i_seg, 1);
            seg_dur = annotated_segments(i_seg, 3);
            seg_offset = seg_onset + seg_dur;
            seg_pitchtrack = cell2mat(cutpitch(seg_onset, seg_offset, file_pitchtrack));
            CurLabel = label_vector(i_seg);

            % Determine the quantums in the segment
            switch CurLabel
                case -1 % transitory
                    num_quantums = 1;
                
                    quantum_onset = seg_onset;
                    quantum_offset = seg_offset; 
                    quantum_dur = seg_dur;
                    quantum_pitchtrack = seg_pitchtrack;


                    dur = quantum_dur;
                    pitchinterval = quantum_pitchtrack(end,2)-quantum_pitchtrack(1,2);
                    fre = 1./(quantum_dur*2);
                    extent = pitchinterval./2;
                    if fre > transitory_cutoff %Hz
                        quantum = struct('file', i_files, 'state', label_to_state(CurLabel), ...
                            'pitchtrack', quantum_pitchtrack, 'onset', quantum_onset, 'offset', quantum_offset, ...
                            'dur', dur, 'pitchinterval', pitchinterval, 'fre', fre, 'extent', extent);
                        quanta(end+1) = quantum;
                    end

                case 0
                    MinPeakDistance = 0;
                    [peaks troughs extremums] = FindLocalextremumsFn(seg_pitchtrack(:,2),seg_pitchtrack(:,1),MinPeakDistance,1);
                    extremums_time = extremums(:,1);
                    num_quantums = size(extremums,1)-1;

                    for i_quantum = 1:num_quantums
                        % Extract features and details for each quantum
                        quantum_onset = extremums_time(i_quantum);
                        quantum_offset = extremums_time(i_quantum+1);
                        quantum_dur = quantum_offset - quantum_onset;
                        quantum_pitchtrack = cell2mat(cutpitch(quantum_onset, quantum_offset, seg_pitchtrack));

                        dur = quantum_dur;
                        pitchinterval = (quantum_pitchtrack(end,2)-quantum_pitchtrack(1,2));
                        fre = 1./(quantum_dur*2);
                        extent = pitchinterval./2;

                        if fre < steady_cutoff %Hz
                       
                            % Assign quantum details in a more concise manner
                            quantum = struct('file', i_files, 'state', label_to_state(CurLabel), ...
                                'pitchtrack', quantum_pitchtrack, 'onset', quantum_onset, 'offset', quantum_offset, ...
                                'dur', dur, 'pitchinterval',pitchinterval,'fre', fre, 'extent', extent);
                            quanta(end+1) = quantum;
                        end
                       
                    end
                case 1
                    MinPeakDistance = 0;
                    [peaks troughs extremums] = FindLocalextremumsFn(seg_pitchtrack(:,2),seg_pitchtrack(:,1),MinPeakDistance,1);
                    extremums_time = extremums(:,1);
                    num_quantums = size(extremums,1)-1;

                    for i_quantum = 1:num_quantums
                        % Extract features and details for each quantum
                        quantum_onset = extremums_time(i_quantum);
                        quantum_offset = extremums_time(i_quantum+1);
                        quantum_dur = quantum_offset - quantum_onset;
                        quantum_pitchtrack = cell2mat(cutpitch(quantum_onset, quantum_offset, seg_pitchtrack));

                        dur = quantum_dur;
                        pitchinterval = (quantum_pitchtrack(end,2)-quantum_pitchtrack(1,2));
                        fre = 1./(quantum_dur*2);
                        extent = pitchinterval./2;

                        if fre < vibrato_cutoff % Hz

                            % Assign quantum details in a more concise manner
                            quantum = struct('file', i_files, 'state', label_to_state(CurLabel), ...
                                'pitchtrack', quantum_pitchtrack, 'onset', quantum_onset, 'offset', quantum_offset, ...
                                'dur', dur, 'pitchinterval',pitchinterval,'fre', fre, 'extent', extent);
                            quanta(end+1) = quantum;
                        end
                        

                    end
                case -2 % no quantum segmentation
                    num_quantums = 1;
                
                    quantum_onset = seg_onset;
                    quantum_offset = seg_offset; 
                    quantum_dur = seg_dur;
                    quantum_pitchtrack = seg_pitchtrack;

                    dur = [];
                    pitchinterval = [];
                    fre = [];
                    extent = [];
                    
                    quantum = struct('file', i_files, 'state', label_to_state(CurLabel), ...
                        'pitchtrack', quantum_pitchtrack, 'onset', quantum_onset, 'offset', quantum_offset, ...
                        'dur', dur, 'pitchinterval',pitchinterval,'fre', fre, 'extent', extent);
                    quanta(end+1) = quantum;
            end
        end

    end

    for i_quantum = 1:length(quanta)

        state = quanta(i_quantum).state;
        quantum_feature = [quanta(i_quantum).dur quanta(i_quantum).extent];
        %quantum_feature = [log(quanta(i_quantum).fre) quanta(i_quantum).extent];
            % Update features structure
            if ~strcmp(label_to_state(CurLabel),'noise')
                if isfield(features, state)
                    features.(state) = [features.(state); quantum_feature];
                else
                    features.(state) = quantum_feature;
                end
            end
        end
    
        %%%%%%%%%%%% add transition mode between quantums based on pitch continuity%%%%
        % transition mode: 1 means current can transit to the next, 0 means
        % no
        % 寻找所有 'file' 字段值为 i_files 的 quantum
    for i_files = 1:length(data)
        
        indices = find([quanta.file] == i_files);
        N_quantum = length(indices);
        for i_quantum = 1:N_quantum-1
            
            if strcmp( quanta(indices(i_quantum)).state,'noise') | strcmp( quanta(indices(i_quantum+1)).state,'noise')
                quanta(indices(i_quantum)).transition = 0;
            else
         

                CurOffset = quanta(indices(i_quantum)).offset;
                NextOnset = quanta(indices(i_quantum+1)).onset;

                % get the onset and offset of each gap between two segments
                GapOnset = CurOffset;
                GapOffset = NextOnset;
                pitchtrackOri = [data(i_files).OriPitch.time data(i_files).OriPitch.pitch];
                Gapdur = GapOffset - GapOnset;
                switch sign(Gapdur)
                    case 1 % Gapdur>0

                        % get the pitch track in the gap region
                        GapPitch = cutpitch(GapOnset,GapOffset,pitchtrackOri);
                        GapPitch = cell2mat(GapPitch);
                        % check if pitchtrack is continous
                        if size(GapPitch,1)>1
                            Gap_t = GapPitch(:,1);
                            if any(diff(Gap_t)>0.006) % there is a silence
                                quanta(indices(i_quantum)).transition = 0;
                            else
                                quanta(indices(i_quantum)).transition = 1;

                            end
                        else
                            quanta(indices(i_quantum)).transition = 1;
                        end
                       
                    otherwise % Gapdur<=0
                        quanta(indices(i_quantum)).transition = 1;
                end
            end
        end
        quanta(indices(end)).transition = 0;

    end
end





    %%%%%%%%%%%% feature extraction %%%%%%%%%%%%%%%%%%%

%% Function to estimate observation likelihood distribution
function [observation_likelihoods_Distr] = estimate_observation_likelihoods(features,stateName,maxComponents)
    
    
    %%%%%%%%%%%% fit GMM %%%%%%%%%%%%%%%%%%%
    num_states = length(stateName);
    observation_likelihoods_Distr = cell(1, num_states);
    
    
    for i = 1:num_states
        % 初始化一个数组用于存储state=2的所有features
        state_features = [];
        
        state_features = features.(stateName{i});
        % 通过BIC 找到最佳numComponents
        [bestGMM, numComponents] = selectBestGMM(state_features, maxComponents);
        k = 10;
        if numComponents == maxComponents
            [bestGMM, bestNumComponents, scores] = kFoldValidationWithPlot(state_features, maxComponents, k)
        end


        % Estimate distribution (e.g., Gaussian mixture model)
        observation_likelihoods_Distr{i} = bestGMM;
        %observation_likelihoods_Distr{i} = fitgmdist(state_features, numComponents)
    end
    %%%%%%%%%%%% fit GMM %%%%%%%%%%%%%%%%%%%


end

% function used to select the best numComponents of GMM
function [bestGMM, numComponents] = selectBestGMM(data, maxComponents)
    % Initialize variables
    bestBIC = Inf;
    bestGMM = [];
    numComponents = 0;
    
    % Loop over the range of component numbers
    for k = 1:maxComponents
        try
            % Fit GMM to data
            gmModel = fitgmdist(data, k, 'RegularizationValue', 0.1);
    
            % Check if the BIC of this model is lower than the best found so far
            if gmModel.BIC < bestBIC
                bestBIC = gmModel.BIC;
                bestGMM = gmModel;
                numComponents = k;
            end
        catch
            % In case fitting fails for a given number of components
            % Do nothing and move to the next number of components
        end
    end
    
    % Check if a valid model was found
    if isempty(bestGMM)
        error('Failed to fit a Gaussian Mixture Model to the data.');
    end
end



%% Function to estimate transition probabilities
function transition_matrix = estimate_transition_matrix(quanta,stateNames,labels)

    % 初始化转移计数矩阵
    transition_counts = zeros(3, 3);
    state_to_label = containers.Map(stateNames, labels);

    for i = 1:length(quanta)-1
        % i
        tran = quanta(i).transition;
        Curstate = quanta(i).state;
        CurLabel = state_to_label(Curstate);
        Nextstate = quanta(i+1).state;
        NextLabel = state_to_label(Nextstate);

        if tran == 1
            row = CurLabel + 2; % 将-1, 0, 1映射到1, 2, 3
            col = NextLabel + 2; % 同上
            transition_counts(row, col) = transition_counts(row, col) + 1;
        end
    end



    % 计算转移概率矩阵
    transition_matrix = transition_counts ./ sum(transition_counts, 2);



end