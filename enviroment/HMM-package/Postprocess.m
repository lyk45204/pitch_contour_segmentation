%%

function [Notes, loc_states] = Postprocess(pitchtrack_piece, inputSampleRate, pruneThresh, stepSize,...
                                   noteout_piece, onsetSensitivity, m_level_piece, ind_silence)
% Postprocess: 1.to use loudness change to determine onset not detected just
% by pitch; 2.get notes and the start time of the states of them and
% disgard short notes.
% input: pitchtrack_piece, inputSampleRate, pruneThresh, stepSize,
%           noteout_piece, onsetSensitivity, m_level_piece, ind_silence,
%           ind_stable(the index of stable in statesname),ind_release
% output: Notes: onset_time, offset_time, medianPitch
%         loc_states: loc_onset, loc_stablestart, loc_releasestart,
%         loc_offset. In them, the frame with value 0 means this note does
%         not have such state.
  

    onsetFrame = 0;
    isVoiced = 0;
    oldIsVoiced = 0;
    nFrame = size(pitchtrack_piece, 1);
    timestamp = pitchtrack_piece(:, 1);
    minNoteFrames = (inputSampleRate*pruneThresh) / stepSize;
    
    notePitchTrack = []; % collects pitches for one note at a time
    onset_time = []; % collects onsets of this piece
    loc_onset = []; % collects the loc of onsets of this piece
    offset_time = []; % collects offsets of this piece
    loc_offset = []; % collects the loc of offsets of this piece
%     loc_stablestart = []; % collects the loc of stablestart of this piece
%     loc_releasestart = []; % collects the loc of releasestart of this piece
    medianPitch = []; % collects the median pitch in notes
    for iFrame = 1:nFrame
        
        isVoiced = noteout_piece(iFrame, 3) < ind_silence && ...
                            pitchtrack_piece(iFrame, 2) > 0 && ...
                            (iFrame >= nFrame-2 || ((m_level_piece(iFrame,2)/m_level_piece(iFrame+2,2)) > onsetSensitivity));
        
        if isVoiced == 1 && iFrame ~= nFrame % voiced frame
        
            if oldIsVoiced == 0 % beginning of a note
            
                onsetFrame = iFrame;
                loc_onset = [loc_onset; onsetFrame];
                onset = timestamp(onsetFrame);
                onset_time = [onset_time; onset];                
            
            end
            
            
            pitch = pitchtrack_piece(iFrame, 2);
            notePitchTrack = [notePitchTrack; pitch]; % add the pitch of current voiced frame to the note's pitch track
            
        else % not currently voiced
            if oldIsVoiced == 1 % the end of note
                
                loc_offset = [loc_offset; iFrame];
                offset = timestamp(iFrame);
                offset_time = [offset_time; offset];
                    
                if length(notePitchTrack) >= minNoteFrames                
                    medianPitch = [medianPitch; median(notePitchTrack)]; % in frequency
                else
                    loc_onset(end) = [];
                    onset_time(end) = [];
                    loc_offset(end) = [];
                    offset_time(end) = [];
                                                                          
                end
                
                notePitchTrack = [];
            end
        end
        oldIsVoiced = isVoiced;
    end
    
    Notes = [onset_time, offset_time, medianPitch];
    
          
end
