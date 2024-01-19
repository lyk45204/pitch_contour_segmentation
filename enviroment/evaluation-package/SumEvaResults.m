function evalMeasures = SumEvaResults(R,outputlevel)
switch outputlevel
    case 'Framelevel'
        evalMeasures(1)=mean([R(:).Precision]);
        evalMeasures(2)=mean([R(:).Recall]);
        evalMeasures(3)=mean([R(:).Fmeasure]);
    case 'Notelevel'
        %%
        evalMeasures(1)=sum([R(:).Dur_GT]);
        evalMeasures(2)=sum([R(:).Dur_TR]);
        evalMeasures(3)=sum([R(:).N_GT]);
        evalMeasures(4)=sum([R(:).N_TR]);
        evalMeasures(5)=length([R(:).COnPOff_listgt]);
        evalMeasures(6)=mean([R(:).COnPOff_Precision]);
        evalMeasures(7)=mean([R(:).COnPOff_Recall]);
        evalMeasures(8)=mean([R(:).COnPOff_Fmeasure]);
        
        evalMeasures(9)=length([R(:).OBOn_listgt]);
        evalMeasures(10)=mean([R(:).OBOn_rategt]);
        evalMeasures(11)=length([R(:).OBOff_listgt]);
        evalMeasures(12)=mean([R(:).OBOff_rategt]);
        
        evalMeasures(13)=length([R(:).S_listgt]);
        evalMeasures(14)=mean([R(:).S_rategt]);
        R_Sratio = [R(:).S_ratio];
        R_Sratio(R_Sratio==0) = [];
        evalMeasures(15)=mean(R_Sratio);
        
        evalMeasures(16)=length([R(:).M_listgt]);
        evalMeasures(17)=mean([R(:).M_rategt]);
        R_Mratio = [R(:).M_ratio];
        R_Mratio(R_Mratio==0) = [];
        evalMeasures(18)=mean(R_Mratio);
        
        evalMeasures(19)=length(vertcat(R(:).PU_listtr));
        evalMeasures(20)=mean([R(:).PU_ratetr]);
        evalMeasures(21)=length([R(:).ND_listgt]);
        evalMeasures(22)=mean([R(:).ND_rategt]);
        
        % time deviation error
        evalMeasures(23)=length([R(:).DVonset_listgt]);
        evalMeasures(24)=mean([R(:).DVonset_rategt]);
        evalMeasures(25)=length([R(:).DVonset_listtr]);
        evalMeasures(26)=mean([R(:).DVonset_ratetr]);
        
        evalMeasures(27)=length([R(:).DVoffset_listgt]);
        evalMeasures(28)=mean([R(:).DVoffset_rategt]);
        evalMeasures(29)=length([R(:).DVoffset_listtr]);
        evalMeasures(30)=mean([R(:).DVoffset_ratetr]);
        
        evalMeasures(31)=length([R(:).DVnote_listgt]);
        evalMeasures(32)=mean([R(:).DVnote_rategt]);
        evalMeasures(33)=length([R(:).DVnote_listtr]);
        evalMeasures(34)=mean([R(:).DVnote_ratetr]);
end
