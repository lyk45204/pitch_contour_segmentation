function data = PitchVersionSelectFn(data,pitchversion)
% input: 'OriPitch' 'InterpoPitch' 'SmoothPitch' 'SmoothInterpoPitch' 'InterpoSmoothPitch'
global data
datalength = length(data);

for i = 1:datalength
    Fnames = fieldnames(data(i));
    IndexFnames = find( strcmp(Fnames,pitchversion)==1 );
    data(i).F0time = data(i).(Fnames{IndexFnames}).time;
    data(i).F0pitch = data(i).(Fnames{IndexFnames}).pitch;
    data(i).FsSel = 1/(data(i).F0time(2) - data(i).F0time(1));
end