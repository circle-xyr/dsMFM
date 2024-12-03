function [FC_sim,metastable_sim,synchrony_sim,FC_cor] = estimation_corr_emp_sim_noRSN(FC_emp,BOLD_sim)

[metastable_sim,synchrony_sim] = BOLD_metastable(BOLD_sim);
FC_mask = tril(ones(size(FC_emp,1),size(FC_emp,1)),0);
y = FC_emp(~FC_mask); %use the elements above the maiin diagnal, y becomes a vector {samples x 1} 
FC_sim=corrcoef(BOLD_sim');
y_sim=FC_sim(~FC_mask);
FC_cor= corr(atanh(y_sim),atanh(y)); %correlation between 2 FCs

%RSN_FC_sim = RSN(BOLD_sim);
%RSN_FC_mask= tril(ones(size(RSN_FC_emp,1),size(RSN_FC_emp,1)),0);
%y_rsn= RSN_FC_emp(~RSN_FC_mask); %use the elements above the maiin diagnal, y becomes a vector {samples x 1} 
%y_rsn_sim = RSN_FC_sim(~RSN_FC_mask);
%RSN_FC_cor = corr(atanh(y_rsn_sim),atanh(y_rsn)); %correlation between 2 FCs
end
