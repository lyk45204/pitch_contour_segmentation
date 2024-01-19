%%%%%%%calculate pIsPitched%%%%%%%%%
function pIsPitched = pvoiced(pitchProb_c_prob)
%pitchProb_c_oneframe is a array, every row is a frame, every column is all the candidate of this frame
%the value is probablity
    m = size(pitchProb_c_prob,1);
    n = size(pitchProb_c_prob,2);
    pIsPitched = zeros(m,1); % it is the prior likelihood of a frame being voiced, constant
    for iCandidate = 1:n 
        pIsPitched = pIsPitched + pitchProb_c_prob(:,iCandidate);
    end
    priorPitchedProb = 0.7;
    priorWeight = 0.5;
    pIsPitched = pIsPitched * (1-priorWeight) + priorPitchedProb * priorWeight; 
end