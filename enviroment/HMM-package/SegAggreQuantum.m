function [ segs] = SegAggreQuantum( decodedSeq,Quantums,state )
%NoteAggreBaseline Note aggregation method for baseline
%Any MIDI change will be considered as a note change
%   Input
%   @decodedSeq: the decoded sequence in index of a specific state number
%   @Quantums: with information of onset, fre, extent, etc.
%   Output
%   @SegsTotal: the aggregated segments in cell arrarys.
%   SegsTotal{1,numCol}: [onset offset duration] 

decodedeSeqDiff = diff([0;decodedSeq;0]);
OnsetMark  = [find(decodedeSeqDiff==1)];
OffsetMark  = [find(decodedeSeqDiff==-1)]-1;
% if decodedSeq(end)==1 % means the last segmentation just has one quantum 
%     OffsetMark = [OffsetMark;OnsetMark(end)];
% end
segs = zeros(length(OnsetMark),3);  %[start time(s); midi NN; duration(s)]

for i = 1:length(OnsetMark)
    idx_b = OnsetMark(i);
    idx_e = OffsetMark(i);
    segs(i,1) = Quantums.onset(idx_b);
    segs(i,2) = Quantums.onset(idx_e)+Quantums.dur(idx_e);
    segs(i,3) = segs(i,2)-segs(i,1);
end


    
    


