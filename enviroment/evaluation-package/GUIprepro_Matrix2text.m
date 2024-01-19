% GUIprepro_Matrix2text: save the matrix into the standard form for the toolbox(only keep the first three digits)
% input: matrix A, the filename
% output: the file in wh
function [] = GUIprepro_Matrix2text(A,filename)

% rearrange the matrix to a column 
A = A'; 
A = A(:);

% save it into txt
fid=fopen(filename,'w');
fprintf(fid,'%6.3f %6.3f %6.3f\n', A);

end