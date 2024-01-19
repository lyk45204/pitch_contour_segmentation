% GUIprepro_Matrix2midi: to save the matrix into midi
% input: matrix A, filename
% output: midi file
function [] = GUIprepro_Matrix2midi(A, filename)
% initialize matrix:
N = size(A,1);  % number of notes
M = zeros(N,6);

M(:,1) = 1;         % all in track 1
M(:,2) = 1;         % all in channel 1
M(:,3) = A(:,3);      % note numbers: one ocatave starting at middle C (60)
M(:,4) = round(linspace(80,120,N))';  % lets have volume ramp up 80->120
M(:,5) = A(:,1);  % note on:  notes start every .5 seconds
M(:,6) = A(:,2);   % note off: each note has duration .5 seconds

midi_new = matrix2midi(M);
writemidi(midi_new, filename);

end