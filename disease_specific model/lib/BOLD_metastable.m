function [metastable,synchrony] = BOLD_metastable(TimeSeries)

      

     N=size(TimeSeries,1);
     for seed=1:N
        Xanalytic = hilbert(demean(TimeSeries(seed,:)));
        Phasesdata(seed,:) = angle(Xanalytic);
     end
       [metastable,synchrony] = phase_metastable(Phasesdata);
       
end       