function FCD_mat = XYR_CBIG_pMFM_step2_generate_FCD_desikan(TC,window)
%计算FCD矩阵
% input：
% TC：时间序列
%window：窗口长度
% This function is the wrapper to generate parcellated time serises and FCD
% for Desikan parcellation
%
% There is no input for this function as it can automatically get the
% output file from previous step.
% There is no output for this function as it will generate the output files
%
% Written by Kong Xiaolu and CBIG under MIT license: https://github.com/ThomasYeoLab/CBIG/blob/master/LICENSE.md


% TC_68 = TC;
% TC_68([1,5,37,41],:) = [];
% FCD_mat = FCD_plot(TC_68, 83);
[dimension,timeline] = size(TC);
mask_tril = ~tril(ones(dimension,dimension));

for i = 1:timeline-window+1
    corr_swc = corrcoef(TC(:,i:i+window-1)');
    corr_vect = corr_swc(mask_tril);
    corr_mat(:,i) = corr_vect;
end
% FCD_mat = corr(corr_mat);
FCD_mat = corr(atanh(corr_mat));
end




