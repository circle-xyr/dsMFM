clear all
close all
%% setup directory for lib, data, cluster, save
main_dir = pwd;
% data_dir = fullfile('F:\XYR\Jean_420\step1_estimation\data_adni_group_avg_34fcd_6p2');
% save_dir_input = fullfile(main_dir,'\2401\test');
data_dir = fullfile('\step1_estimation\test');
save_dir_input = fullfile(main_dir,'6p2_34fc_onlyAtrophy\100');
save_dir = save_dir_input;

SC_dir = fullfile('\step1_estimation\data_adni_low');
a=load([SC_dir '\ADNI_CN_ave.mat' ]);
SC=a.SC;
SC = SC./max(max(SC)).*0.2;

cd('..');
high_dir = pwd;
lib_dir = fullfile(high_dir, 'lib');
addpath(lib_dir);

list = dir(fullfile(data_dir,'*.mat'));
fileNums = length(list);

c=load('\group_avg_34fcd.mat');


%--------------------------------------------------------------------------

for cishu=1:100
    for sub = 1 : fileNums
        
        load([data_dir,'\',list(sub).name ],'Para_E','FC','Myelin','atrophy');
        FC_emp=FC;
        C1 = strsplit(list(sub).name,{' ','_','.'},'CollapseDelimiters',true);
        filen = num2str(C1{8});
        if (strcmp(C1{8},'MCI'))
      
            y2 = c.fcdpdf_MCI;
            Tmax=9.35;
            TR=3;
          
            Nstate = rng;
            BOLD_TR = XYR_MFMem_rfMRI_nsolver_eul_sto_timeserice_6p2(Para_E,SC,Nstate,Tmax,TR,Myelin,atrophy);
            [FC_sim,metastable_sim,synchrony_sim,FC_cor] = estimation_corr_emp_sim_noRSN(FC_emp,BOLD_TR);
            fcd = XYR_CBIG_pMFM_step2_generate_FCD_desikan(BOLD_TR,34);
            [FCDpdf,binEdges,FCDpdf2] =XYR_FCDpdf(fcd);
            ks_check = XYR_kstest2_pdf(FCDpdf,y2);
            [ssim1,ssim2]=ssim(FC_sim,FC_emp);
           
            
            
            if exist(save_dir )==0
                mkdir(save_dir );
            end
            save( [save_dir ,'\',num2str(cishu),'_para_', num2str(C1{4}),'_', num2str(C1{5}),'_data_adni_low_',filen] ,'ssim1','FC_sim','metastable_sim','synchrony_sim','FC_cor','BOLD_TR','fcd','FCDpdf','binEdges','ks_check');
            disp([ num2str(cishu) '_' num2str(C1{4}) '_' num2str(C1{8}) '_finish']);
            %         clear myleincn myelin
        end
        
        if (strcmp(C1{8},'AD'))
            
            y2 = c.fcdpdf_AD;
            Tmax=9.35;
            TR=3;
            %         step=1;
            %         ks_check = 1;
            %         while ks_check>0.3&&step<100
            Nstate = rng;
            BOLD_TR = XYR_MFMem_rfMRI_nsolver_eul_sto_timeserice_6p2(Para_E,SC,Nstate,Tmax,TR,Myelin,atrophy);
            [FC_sim,metastable_sim,synchrony_sim,FC_cor] = estimation_corr_emp_sim_noRSN(FC_emp,BOLD_TR);
            fcd = XYR_CBIG_pMFM_step2_generate_FCD_desikan(BOLD_TR,34);
            [FCDpdf,binEdges,FCDpdf2] =XYR_FCDpdf(fcd);
            ks_check = XYR_kstest2_pdf(FCDpdf,y2);
            [ssim1,ssim2]=ssim(FC_sim,FC_emp);
         
            
            if exist(save_dir )==0
                mkdir(save_dir );
            end
            save( [save_dir ,'\',num2str(cishu),'_para_', num2str(C1{4}),'_', num2str(C1{5}),'_data_adni_low_',filen] ,'ssim1','FC_sim','metastable_sim','synchrony_sim','FC_cor','BOLD_TR','fcd','FCDpdf','binEdges','ks_check');
            disp([ num2str(cishu) '_' num2str(C1{4}) '_' num2str(C1{8}) '_finish']);
            %         clear myleincn myelin
        end
        
    end
end

rmpath(lib_dir);
disp('finish');

heatmap(Z1);