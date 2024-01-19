%% MonoNote
% input: nSPP, nPPS, minPitch, path_piece, pitchtrack_piece
% output: notes: Frametime, decoded pitch and the state of every frame
%              
function noteout = MonoNote(nSPP, nPPS, minPitch, path_piece, pitchtrack_piece)

%%  calculate noteout_temp containing time, pitch, state of every frame
currPitch = floor((path_piece-1)/nSPP)*(1/nPPS)+minPitch;
% currPitch = s2f(currPitch); %transfer semitone to frequency 
stateKind = rem((path_piece-1),nSPP)+1; %// unvoiced, attack, stable, release, inter
currPitch(find(stateKind==nSPP)) = NaN; %make the silence having no pitch

t_my = pitchtrack_piece(:,1); %time
nFrame = length(path_piece);
noteout = zeros(nFrame,3);
noteout(:,1) = t_my;
noteout(:,2) = currPitch;
noteout(:,3) = stateKind;

end