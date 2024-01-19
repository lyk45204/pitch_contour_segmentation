function pitchtrackOutput = cutpitch(Vonset,Voffset,pitchtrackInput)
% This function use two timing points, onset and offset to cut a pitch
% track
% input:
%   @ onset
%   @ offset
%   @ pitchtrackInput:[time,pitch]
% output:
%   @pitchtrackOutput:cell{[time,pitch];[time,pitch];[time,pitch]}

time = pitchtrackInput(:,1);
pitch = pitchtrackInput(:,2);
N_seg = length(Vonset);
pitchtrackOutput = cell(N_seg,1);

for i_seg = 1:N_seg
    onset = Vonset(i_seg);
    offset = Voffset(i_seg);
    [v1,idx1] = min( abs(time-onset) );
    [v2,idx2] = min( abs(time-offset) );
    pitchtrackOutput{i_seg} = [time(idx1:idx2),pitch(idx1:idx2)];
end
    
