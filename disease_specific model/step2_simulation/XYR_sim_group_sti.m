clear all
close all

%% setup directory for lib, data, cluster, save

main_dir = pwd;
data_dir = fullfile('step1_estimation\result_data_adni_low_group_avg_34fc_disc_6p2');
save_dir_input = fullfile(main_dir,'\result_group_sti_0-1_0.02_f10_90_120_34fc_6p2');
% save_dir_input = fullfile(main_dir,'\2309\result_mulpoint');
save_dir = save_dir_input;

SC_dir = fullfile('\step1_estimation\data_adni_low');
a=load([SC_dir '\ADNI_CN_ave.mat' ]);
SC=a.SC;
SC = SC./max(max(SC)).*0.2;

% FC_nc_dir = fullfile('F:\XYR\Jean_420\step1_estimation\data_adni_low_x');
% a=load([FC_nc_dir '\ADNI_CN_ave' ]);
FC_nc=a.FC_emp;
cd('..');
high_dir = pwd;
lib_dir = fullfile(high_dir, 'lib');
addpath(lib_dir);

list = dir(fullfile(data_dir,'*.mat'));
fileNums = length(list);
c=load('group_avg_34fcd.mat');
% myelin_dir = 'D:\yan\data\ADNI_myelin\MCI_result\';
y1= c.fcdpdf_CN;

%--------------------------------------------------------------------------
for cishu=1:1
    for sub = 3:3
        % for sub = 3
        C1 = strsplit(list(sub).name,{' ','_','.'},'CollapseDelimiters',true);
        filen = num2str(C1{8});
        load([data_dir,'\',list(sub).name ],'Para_E','FC','Myelin','atrophy');
        FC_emp=FC;
        
        
        Tmax=9.35;
        TR=3;
        %find out number of brain regions
        NumC = length(diag(SC));
        
        %% prepare the model
        
        for stil=1:68
            
            %
            for i=1:1%NumC
                
                B=0.005;   %s
                TstiStart=90;%s stimulation start time
                TstiCon=120;%s stimulation continue time
   
                for A=0.02:0.02:1
  
          
                    for f=10:10
                        %                             if f~=0
                        if exist([save_dir,'\',num2str(A) '_' num2str(f) '_' num2str(B) '_' num2str(TstiCon) '\',num2str(cishu)])==0
                            mkdir([save_dir,'\',num2str(A) '_' num2str(f) '_' num2str(B) '_' num2str(TstiCon) '\',num2str(cishu)]);
                        end
                        t=TstiStart:0.001:TstiStart+TstiCon;%%%%%%%
                        I =-A*XYR_H(sin(2*pi*f*t)).*(1-XYR_H(sin(2*pi*f*(t+B))));
                        [BOLD_TR] = XYR_MFMem_rfMRI_nsolver_eul_sto_timeserice_6p2_sti2(Para_E,SC,Nstate,Tmax,TR,stil,A,B,f,TstiStart,TstiCon,Myelin,atrophy);
                        
                        %FC corr
                        [FC_sim,metastable_sim,synchrony_sim,sFC_cor,nFC_cor] = estimation_corr_emp_sim_noRSN2(FC_emp,FC_nc,BOLD_TR);
                        %FCDpdf ks
                        fcd = XYR_CBIG_pMFM_step2_generate_FCD_desikan(BOLD_TR,34);
                        [FCDpdf,binEdges,pdf] =XYR_FCDpdf(fcd);
                        
                        ks_check_n = XYR_kstest2_pdf(FCDpdf,y1);
                        
                        
                        save( [save_dir,'\',num2str(A) '_' num2str(f) '_' num2str(B) '_' num2str(TstiCon),'\',num2str(cishu), '\ADNI_' num2str(i) '_sti_' num2str(stil) '_' filen] ,'FC_sim','metastable_sim','synchrony_sim','sFC_cor','nFC_cor','BOLD_TR','I','fcd','FCDpdf','ks_check_n');
                        disp([num2str(cishu) 'sub' num2str(sub) '_A:' num2str(A) '_f:' num2str(f) '_sti:' num2str(stil) '_finish']);
                        %                             end
                    end
                end
            end
        end
    end
    
end

disp('finish');

