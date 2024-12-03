function [metastable,synchrony] = phase_metastable(Phases_Save)
%% Compare wimth metastable ,synchrony

  N = size(Phases_Save,1); 
  Tmax= size(Phases_Save,2);
 
    
 for t=1:Tmax
     kudata=sum(complex(cos(Phases_Save(:,t)),sin(Phases_Save(:,t))))/N;
     op(t)=abs(kudata);
 end
 synchrony=mean(op);
 metastable=var(op);
end
 
