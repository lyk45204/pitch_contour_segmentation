% read data

clc; clear all; close all;
set(groot, 'defaultAxesFontSize', 18);
set(groot, 'defaultTextFontSize', 18);
set(groot, 'DefaultLineLineWidth', 2); % Sets the default line width to 2

% set parameters by observation from the quantum distribution
%% readAudio
% test
% filePathRoot = ['/Users/yukunli/Desktop/PhD/Projects/Quantum of pitch/Dataset/SMC2016_noteseg/SMC2016-master/samllDataset/Test/'];
% train
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

%% save data
% testData = data;
trainData = data;
% save('testData.mat', 'testData');
save('trainData.mat', 'trainData');

