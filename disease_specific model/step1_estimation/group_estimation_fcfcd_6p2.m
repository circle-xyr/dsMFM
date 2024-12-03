clear all
close all

%% input parameters defined by users
% N_core: Number of CPU cores for calculation
% EstimationMaxStep: Number of estimation setp
% save_dir_input: Output file directory

N_core = 4; % user define
EstimationMaxStep = 500; % user define
poolobj = gcp('nocreate');
delete(poolobj);

%% setup directory for lib, data, cluster, save
main_dir = pwd;
SC_dir = fullfile(main_dir,'data_adni_low');
% FC_dir = fullfile(main_dir,g);
save_dir_input = fullfile(main_dir,'result');
save_dir =fullfile(main_dir,'result_data_adni_low_group_avg_34fc_disc_6p2');
cluster_dir = save_dir_input;

cd('..');
high_dir = pwd;
lib_dir = fullfile(high_dir, 'lib');
addpath(lib_dir);


%--------------------------------------------------------------------------
%% setup working conditions
%-------------------------
%CPU-Cores for caculation
%-------------------------

%-------------------------
%maximum Estimation steps
%------------------------

%--------------------------------------------------------------------------

%% some important notation  ->

%for emprical data:
%y:    empricial data  {nT x 1}
%n:    number of FC are used
%T:    samples of each FC
%nT:   number of total empricial data samples
%NumC: number of brain regions

%for model parameter:
%p:         number of parameters
%Prior_E:   expectation of parameter value {p x 1}
%Para_E:    model parameter {p x 1}
%A:         re-parameter of original model parameter, by A = log(Para_E./Prior_E)
%A_Prior_C: variance matrix of model parameter  {p x p}
%Ce:        estimated noise covariance {nTxnT}
%lembda:    parameter used to approximate the Ce, by
%           for i = 1:n
%                DiagCe(T*(i-1)+1:T*(i-1)+T) = exp(lembda(i));
%           end
%
%for details, reference paper: [2,4]

%for estimation
%h_output:  model output {nT x 1}
%r:         error, difference between y and h_output
%JF:        Jacobian matrix of h_output at Para_E, using Newton-forwards approximation {nT x p}
%JK:        Jacobian same above, using complex-step approximation {nT x p}
%dldpara:   1st. derivative, {p x 1}, used in Gauss-Newton search
%dlddpara:  inv, negative, 2nd. derivative, {p x p}, used in Gauss-Newton search
%
%for data save
%rrr_z:     save the correlation between emprical FC and simulated FC, z-transfered
%
%----------------------------------------------------------
%% prepare the empirical data y
% list=dir(fullfile(data_dir,'*.mat'));
% fileNums=length(list);
% for sub=1:fileNums
% b=load([FC_dir '\FC_emp_avg0.mat']);
% FC = b.FC_emp_avg;

a=load([SC_dir '/ADNI_CN_ave.mat' ]);
SC=a.SC;
SC = SC./max(max(SC)).*0.2;

%
load('atrophy_adni.mat');
atrophy=mean(Wad)';
% atrophy=zeros(68,1);
myelin_dir = 'ADNI_myelin';
Myelin=load(myelin_dir, '\atrophy_adni.mat');

b=load([main_dir '\data_adni_low_x_group_avg' '\ADNI_CN_ave.mat' ]);
FC = b.FC_emp;
c=load('fcd.mat');
window = 34;
%find out number of brain regions
NumC = length(diag(SC));

FC_mask = tril(ones(size(FC,1),size(FC,1)),0);
y = FC(~FC_mask); %use the elements above the maiin diagnal, y becomes a vector {samples x 1}
y2 = c.fcdpdf; %emp fcdpdf
n = 1;            %only one FC
T = length(y);    %samples of FC
nT = n*T;         %number of data samples
TR=3;
Tmax=9.35;
%-----------------------------------------------------------
for run_time = 1:20
    
    %-----------------------------------------------------------
    %% prepare the model parameters
    
    %-----------------------------------------------------------
    % set up prior for G(globle scaling of SC), w(self-connection strength/excitatory),Sigma(noise level),Io(background input)
    p = 3 + 3; %number of estimated parameter
    Prior_E = zeros(p,1);
    
    %--------------------------------
    %basic value / expec3tation value
    %-------------------------------
    Prior_E(1) = 1;%w
    Prior_E(2) = 0.5;%aw
    Prior_E(3) = 0.1;%bw
    Prior_E(4) = 0.32;%I0
    Prior_E(5) = 1; %G
    Prior_E(6) = 0.001;%sigma
    %  G = 2.1780;
    %---------------------------------------------------------
    
    %-------------------------------------------------------------------------
    %Prior for Re-Parameter A,  Parameter_model = Prior_E.*exp(A), A~Normal(E=0,C)
    A_Prior_C = 1/4*ones(1,p);%variance for parameter
    A_Prior_C = diag(A_Prior_C);
    A_Prior_E = zeros(p,1);
    invPrior_C = inv(A_Prior_C);
    %---------------------------------------------------------
    
    %---------------------------------------------------------
    %==========================
    %initial Parameter
    %==========================
    Para_E = Prior_E;
    Para_E_new = Para_E
    
    
    %re-paramter of Para_E
    A = log(Para_E./Prior_E);
    %-----------------------------------------------------------
    rng(run_time)
    A_start = -1/4+(1/4+1/4)*rand(3+3,1);
    A = A_start;
    Para_E = exp(A).*Prior_E;
    Para_E(4) = 0.32;
    Para_E_new = Para_E;
    %<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    %% begin estimation
    
    step = 1; %counter
    
    % setup save vectors
    
    metastable_step = zeros(1,EstimationMaxStep+1);
    synchrony_step = zeros(1,EstimationMaxStep+1);
    CC_check_step = zeros(1,EstimationMaxStep+1);     %save the fitting criterion, here is the goodness of fit, same as rrr below
    lembda_step_save = zeros(n,EstimationMaxStep);    %save the Ce
    rrr = zeros(1,EstimationMaxStep);                 %save the goodness of fit
    rrr_z  = zeros(1,EstimationMaxStep);              %save the correlation between emprical FC and simulated FC, z-transfered
    rrr_d = zeros(1,EstimationMaxStep);
    rrr_fcd  = zeros(1,EstimationMaxStep);
    rrr_total  = zeros(1,EstimationMaxStep);
    ks_check_step =  zeros(1,EstimationMaxStep);
    total_check_step =  zeros(1,EstimationMaxStep);
    Para_E_step_save = zeros(p,EstimationMaxStep);    %save the estimated parameter
    
    %setup the cluster
    cluster = parcluster('local');
    cluster.JobStorageLocation = cluster_dir;
    parpool(cluster,N_core);
    %--------------------------------------start whole loop, begin estimation
    while (step <= EstimationMaxStep)
        %---------------------
        
        
        step
        fix(clock)
        
        Para_E_step = Para_E;
        
        
        if step == 1
            load(fullfile(save_dir_input ,'saved_original_random_generator.mat'),'Nstate') %use the same randon generator in our paper
        else
            Nstate = rng;
        end
        
        
        %<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        %caculation h_output {nT x 1}
        %<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        
        %
        funcP = @(Para_E) XYR_MFMem_rfMRI_nsolver_eul_sto_6p2(Para_E,Prior_E,SC,y,FC_mask,Nstate,Tmax,TR,0,Myelin,atrophy,window);
        funcA = @(A) XYR_MFMem_rfMRI_nsolver_eul_sto_6p2(A,Prior_E,SC,y,FC_mask,Nstate,Tmax,TR,1,Myelin,atrophy,window);
        
        
        [h_output, CC_check ,FC_sim,FCD,metastable_sim,synchrony_sim] = funcP(Para_E);  %CC_check: cross-correlation check of two FCs
        %h_output: output of model, entries above the main diagonal of the simulated FC, z-transfered
        fcd_mask = tril(ones(size(FCD,1),size(FCD,1)),0);
        FCDR = FCD(~fcd_mask);
        [FCDpdf,binEdges,FCDR] =XYR_FCDpdf(FCD);
        ks_check = XYR_kstest2_pdf(FCDpdf,y2);
        FCDpdf=FCDpdf';
        total_check = (1-CC_check) + ks_check;
        
        
        %<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        %caculation of Jacobian, JF, JK, {nT x p }
        %<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        JF = zeros(nT,p);
        JK = JF;
        JFK = [JF JK];
        
        %begin parallel computing to caculate the Jacobian
        %--------------------------------------------------------------------
        parfor i = 1:2*p
            if i <= p
                disp('running JF')
                JFK(:,i) = CBIG_MFMem_rfMRI_diff_P1(funcA,A,h_output,i); % {nT x p}
            else
                disp('running Jk')
                JFK(:,i) = CBIG_MFMem_rfMRI_diff_PC1(funcA,A,i-p);
            end
        end
        %--------------------------------------------------------------------
        %end parallel computiong
        
        JF  = JFK(:,1:p); % {nT x p}
        JK  = JFK(:,p+1:2*p);
        
        %<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        %caculation of r, difference between emprical data y and model output h_output
        %<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        
        r = y - h_output; % {n*T x 1}
        
        
        %<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        %prepare parallel computing of EM-algorithm
        %<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        A_old = A;
        
        A_FDK = zeros(p,2);
        h_output_FDK = zeros(nT,2);
        r_FDK = r;
        lembda_FDK = zeros(n,2);
        
        dlddpara_FDK = zeros(p,p,2);
        dldpara_FDK = zeros(p,2);
        
        
        LM_reg_on = [1 1]; %switcher of Levenberg-Marquardt regulation, started if correlation between FCs > 0.4
        
        %<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        %Estimation using Gauss-Newton and EM begin here, cautions by modification
        %<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        
        %start parallel computing of EM
        %----------------------------------------------------------------------
        parfor ppi= 1:2                  %begin parfor caculation
            
            if ppi == 1   %first ,J = JF
                J = JF;
                r = r_FDK;
                disp('begin J = FD');
            else
                J = JK;
                r = r_FDK;
                disp('begin J = K');
            end
            
            % prepare lembda for lembda, Ce
            
            lembda = -3*ones(n,1);
            DiagCe = ones(1,nT);  %only have main diagonal entries
            for i = 1:n
                DiagCe(T*(i-1)+1:T*(i-1)+T) = exp(lembda(i));
            end
            %inv(Ce):
            inv_DiagCe = DiagCe.^-1;  %try to only use diagonal element
            
            % preparation g, H, for g & H, see [2]
            g = zeros(n,1); %initialization
            H = zeros(n,n); %initialization
            
            
            for mi = 1:16  %<-------------------------------begin M-step loop
                
                %          disp('m step:')
                %          disp(mi)
                
                %-------------------------------------------------------
                %
                % P = inv(Ce) - inv(Ce) * J * pinv(J'*inv(Ce)*J) * J' * inv(Ce); {nT x p}
                %
                % see [2,3]
                %-------------------------------------------------------
                
                %first computing: pinv(J'*inv(Ce)*J)
                inv_JinvCeJ = zeros(p,nT);
                %step1: J'*inv(Ce)
                for i = 1:p
                    inv_JinvCeJ(i,:) = bsxfun(@times,J(:,i)', inv_DiagCe);
                end
                %step2: J'*inv(Ce)*J
                inv_JinvCeJ = inv_JinvCeJ*J;
                %step3: pinv(J'*inv(Ce)*J)
                inv_JinvCeJ = pinv(inv_JinvCeJ);
                
                %now computing:  %inv(Ce) * J * inv_JinvCeJ * J' * invCe
                P = zeros(nT,p);
                %step1: inv(Ce) * J
                for i = 1:p
                    P(:,i) = bsxfun(@times, J(:,i), inv_DiagCe');
                end
                %step2: (inv(Ce) * J) * inv_JinvCeJ * J'
                P = P*inv_JinvCeJ*J';
                %step3:  -(inv(Ce) * J * inv_JinvCeJ * J') * inv(Ce)
                for i = 1:nT
                    P(:,i) = bsxfun(@times, P(:,i), -inv_DiagCe');
                end
                %step4: invCe - (inv(Ce) * J * inv_JinvCeJ * J' * inv(Ce) )
                P(1:(nT+1):nT*nT) = bsxfun(@plus, diag(P)',inv_DiagCe);
                
                P = single(P);   %memory trade off
                
                
                %-------------------------------------------------------
                %
                % g(i) = -0.5*trace(P*exp(lembda(i))*Q(i))+0.5*r'*invCe*exp(lembda(i))*Q(i)*invCe*r;  {n x 1}
                %                         d  Ce
                % exp(lembda(i))*Q(i) =  -- ---
                %                         d  lembda(i)
                %
                % see [2,3]
                %-------------------------------------------------------
                
                for i = 1:n
                    %step1: 0.5*r'*invCe*exp(lembda(i))*Q(i)
                    g(i) = -0.5*exp(lembda(i))*trace(P(T*(i-1)+1:T*(i-1)+T,T*(i-1)+1:T*(i-1)+T));
                    %step2: (0.5*r'*invCe*exp(lembda(i))*Q(i))*invCe*r
                    g_rest = 0.5*bsxfun(@times,r',inv_DiagCe)*exp(lembda(i))*CBIG_MFMem_rfMRI_matrixQ(i,n,T); %CBIG_mfm_rfMRI_matrixQ is used to caculate Q(i)
                    g_rest = bsxfun(@times,g_rest,inv_DiagCe)*r;
                    %step3:
                    g(i) = g(i) + g_rest;
                end
                
                %-------------------------------------------------------
                %
                %H(i,j) = 0.5*trace(P*exp(lembda(i))*Q(i)*P*exp(lembda(j))*Q(j)); {n x n}
                %
                % see [2,3]
                %-------------------------------------------------------
                
                for i = 1:n
                    for j = 1:n
                        Pij = P(T*(i-1)+1:T*(i-1)+T,T*(j-1)+1:T*(j-1)+T);
                        Pji = P(T*(j-1)+1:T*(j-1)+T,T*(i-1)+1:T*(i-1)+T);
                        H(i,j) = 0.5*exp(lembda(i))*exp(lembda(j))*CBIG_MFMem_rfMRI_Trace_AXB(Pij,Pji);
                    end
                end
                
                %clear P Pij Pji
                P = [];
                Pij = [];
                Pji = [];
                
                %update lembda
                d_lembda = H\g; % delta lembda
                
                lembda = lembda + d_lembda;
                
                for i = 1:n
                    if lembda(i) >= 0
                        lembda(i) = min(lembda(i), 10);
                    else
                        lembda(i) = max(lembda(i), -10);
                    end
                end
                
                %--------------------------------------------------------------------------
                
                
                
                %update Ce for E-step
                DiagCe = ones(1,nT);
                for i = 1:n
                    DiagCe(T*(i-1)+1:T*(i-1)+T) = exp(lembda(i));
                end
                inv_DiagCe = DiagCe.^-1;
                
                %abort criterium of m-step
                if max(abs(d_lembda)) < 1e-2, break, end
                
            end
            %<-------------------end M-step loop %----------------------------------
            
            %display lembda
            lembda
            lembda_FDK(:,ppi) = lembda;
            
            
            %----------------E-step-----------------------------------------------
            
            %-------------------------------------------------------------------
            %
            %dldpara:   1st. derivative, {p x 1}, used in Gauss-Newton search
            %           dldpara = J'*inv(Ce)*r + inv(Prior_C)*(A_Prior_E - A);
            %
            %dlddpara:  inv, negativ, 2nd. derivative, {p x p}, used in Gauss-Newton search
            %           dlddpara = (J'*inv(Ce)*J + inv(Prior_C));
            %
            %see [2,3]
            %-------------------------------------------------------------------
            
            JinvCe = zeros(p,nT); %J'invCe
            for i = 1:p
                JinvCe(i,:) = bsxfun(@times,J(:,i)', inv_DiagCe);% J'%invCe <----- p x nT
            end
            
            
            
            dlddpara = (JinvCe*J + invPrior_C); % inv, negativ, von 2nd. derivative {p x p}
            
            dldpara = JinvCe*r + invPrior_C*(A_Prior_E - A); % 1st. derivative, {p x 1}
            
            JinvCe = []; %save the memory
            
            
            d_A = dlddpara\dldpara;
            A_FDK(:,ppi) = A + d_A; %newton-gauss, fisher scoring, update Para_E
            Para_E_new = exp(A_FDK(:,ppi)).*Prior_E;
            Para_E_new(4) = 0.32;
            dPara_E_new = abs(Para_E_new - Para_E);
            
            if any(bsxfun(@ge,dPara_E_new,0.5)) %paramter should not improve too much
                d_A = (dlddpara+10*diag(diag(dlddpara)))\dldpara;
                disp('using reg = 10')
                A_FDK(:,ppi) = A + d_A; %newton-gauss, fisher scoring, update Para_E
                Para_E_new = exp(A_FDK(:,ppi)).*Prior_E;
                Para_E_new(4) = 0.32;
                LM_reg_on(ppi) = 0;
            end
            
            [h_output_FDK(:,ppi), CC_check_FDK(:,ppi),FC_sim,FCD,metastable_FDK(:,ppi),synchrony_FDK(:,ppi)] = funcP(Para_E_new);
            fcd_mask = tril(ones(size(FCD,1),size(FCD,1)),0);
            FCDR_FDK(:,ppi) = FCD(~fcd_mask);
            [FCDpdf,binEdges,FCDR] =XYR_FCDpdf(FCD);
            ks_check_FDK(:,ppi) = XYR_kstest2_pdf(FCDpdf,y2);
            total_check_FDK(:,ppi) = (1-CC_check_FDK(:,ppi)) + ks_check_FDK(:,ppi);
            FCDpdf_FDK(:,ppi)=FCDpdf';
            r = y - h_output_FDK(:,ppi);
            
            dlddpara_FDK(:,:,ppi) = dlddpara;
            dldpara_FDK(:,ppi) = dldpara;
            
            
        end
        %<---------------------------------------------------------------------
        %end parallel computiong
        
        %<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        %comparision the Fitting improvement between using JF and JK, choose the better one
        %<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        
        
        % comprision JF and JK------------------------------------------------------
        F_comparison = total_check_FDK(:,1);
        K_comparison = total_check_FDK(:,2);
        
        if F_comparison >= K_comparison
            A = A_FDK(:,1);
            h_output = h_output_FDK(:,1);
            fcdpdf_output=FCDpdf_FDK(:,1);
            CC_check_step(step+1) = CC_check_FDK(:,1);
            metastable_step(step+1) = metastable_FDK(:,1);
            synchrony_step(step+1) = synchrony_FDK(:,1);
            ks_check_step(step+1) = ks_check_FDK(:,1);
            total_check_step(step+1) = total_check_FDK(:,1);
            lembda_step_save(:,step) = lembda_FDK(:,1);
            dlddpara = dlddpara_FDK(:,:,1);
            dldpara = dldpara_FDK(:,1);
            
            if CC_check_step(step+1) > 0.4   %Levenberg-Marquardt regulation, started if correlation between FCs > 0.4
                LM_on = LM_reg_on(1);
            else
                LM_on = 0;
            end
            
            disp('choose FD')
            
        else
            A = A_FDK(:,2);
            h_output = h_output_FDK(:,2);
            fcdpdf_output=FCDpdf_FDK(:,2);
            CC_check_step(step+1) = CC_check_FDK(:,2);
            metastable_step(step+1) = metastable_FDK(:,2);
            synchrony_step(step+1) = synchrony_FDK(:,2);
            ks_check_step(step+1) = ks_check_FDK(:,2);
            total_check_step(step+1) = total_check_FDK(:,2);
            lembda_step_save(:,step) = lembda_FDK(:,2);
            dlddpara = dlddpara_FDK(:,:,2);
            dldpara = dldpara_FDK(:,2);
            
            if CC_check_step(step+1) > 0.4 %Levenberg-Marquardt regulation, started if correlation between FCs > 0.4
                LM_on = LM_reg_on(2);
            else
                LM_on = 0;
            end
            
            
            disp('choose Komplex')
        end
        
        % -----------------End comparision------------------------------------------------
        
        
        
        %<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        %now adding levenberg-Maquadrat
        %<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        if LM_on == 1
            
            disp('begin Levenberg')
            
            lembda = lembda_step_save(:,step);
            
            DiagCe = ones(1,nT);
            for i = 1:n
                DiagCe(T*(i-1)+1:T*(i-1)+T) = exp(lembda(i));
            end
            inv_DiagCe = DiagCe.^-1;  %try to only use diagonal element
            
            %regulation value table
            reg_reg = [0,1,10,100];
            Nreg = length(reg_reg);
            
            A_reg = zeros(p,Nreg);
            h_output_reg = zeros(nT,Nreg);
            lembda_reg = zeros(n,Nreg);
            
            %transfer results for reg = 0
            A_reg(:,1) = A;
            h_output_reg(:,1) = h_output;
            FCDpdf_reg(:,1) =fcdpdf_output;
            CC_check_reg(:,1) = CC_check_step(step+1);
            metastable_reg(:,1) = metastable_step(step+1);
            synchrony_reg(:,1) = synchrony_step(step+1);
            ks_check_reg(:,1) = ks_check_step(step+1);
            total_check_reg(:,1) = total_check_step(step+1);
            
            %<--------begin parallel computing-------------------------------
            parfor ppi = 2:Nreg
                
                
                reg = reg_reg(ppi);
                A = A_old;
                
                d_A = (dlddpara+reg*diag(diag(dlddpara)))\dldpara; %LM
                A_reg(:,ppi) = A + d_A; %newton-gauss, fisher scoring, update Para_E
                Para_E_new = exp(A_reg(:,ppi)).*Prior_E;
                Para_E_new(4) = 0.32;
                [h_output_reg(:,ppi), CC_check_reg(:,ppi),FC_sim,FCD,metastable_reg(:,ppi),synchrony_reg(:,ppi)] = funcP(Para_E_new);
                fcd_mask = tril(ones(size(FCD,1),size(FCD,1)),0);
                FCDR_reg(:,ppi) = FCD(~fcd_mask);
                [FCDpdf,binEdges,FCDR] =XYR_FCDpdf(FCD);
                ks_check_reg(:,ppi) = XYR_kstest2_pdf(FCDpdf,y2);
                total_check_reg(:,ppi)= (1- CC_check_reg(:,ppi)) +  ks_check_reg(:,ppi);
                FCDpdf_reg(:,ppi)= FCDpdf';
                r = y - h_output_reg(:,ppi);
                
            end
            %<--------------------end parallel computing------------------------------
            
            
            clear DiagCe inv_DiagCe
            
            T_comparision = total_check_reg;
            [CC_check_step_save(step+1),T_comparision_indx] = min(T_comparision);
            
            disp(['chosen reg is: ' num2str(reg_reg(T_comparision_indx))]);
            A = A_reg(:,T_comparision_indx);
            h_output = h_output_reg(:,T_comparision_indx);
            fcdpdf_output = FCDpdf_reg(:,T_comparision_indx);
            CC_check_step(step+1) = CC_check_reg(:,T_comparision_indx);
            metastable_step(step+1) = metastable_reg(:,T_comparision_indx);
            synchrony_step(step+1) = synchrony_reg(:,T_comparision_indx);
            ks_check_step(step+1) = ks_check_reg(:,T_comparision_indx);
            total_check_step(step+1) = total_check_reg(:,T_comparision_indx);
            
        end
        %--------------------------------------------------------------------------------
        
        
        
        %<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        %update results, check abbort criterium
        %<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
        
        Para_E = exp(A).*Prior_E;
          Para_E(4) = 0.32;
        d_Para_E = Para_E - Para_E_step;
        
        dN = sqrt(sum(d_Para_E.^2))
        
        rrr(step) = 1-(var(y-h_output)/var(y));%goodness of fit
        rrr_z(step)  = corr(atanh(h_output),atanh(y)); %correlation between 2 FCs
        rrr_d(step) = 1-(var(y2-fcdpdf_output')/var(y2));%goodness of fit
        rrr_fcd(step) =  XYR_kstest2_pdf(fcdpdf_output',y2);
        rrr_total(step) = (1-rrr_z(step)) + rrr_fcd(step);
        
        
        Para_E_step_save(:,step) = Para_E;
        Para_E
        disp(['step of fitting FC = ' num2str(rrr(step))])
        disp(['step of fitting correlation FC = ' num2str(rrr_z(step))])
        disp(['step of fitting FCD = ' num2str(rrr_d(step))])
        disp(['step of fitting correlation FCD = ' num2str(rrr_fcd(step))])
        disp(['step of M = ' num2str( metastable_step(step))])
        disp(['step of S = ' num2str( synchrony_step(step))])
        disp('---------------------')
        
        [rrr_temp,index_temp] = min(rrr_total(rrr_total~=0));
        disp(['best of total = ' num2str(rrr_temp)])
        disp(['best of fitting correlation FC = ' num2str(rrr_z(index_temp))])
        disp(['best of fitting correlation FCD = ' num2str(rrr_fcd(index_temp))])
        disp(['best of M = ' num2str( metastable_step(index_temp))])
        disp(['best of S = ' num2str( synchrony_step(index_temp))])
        %Abort criterium of total estimation
        if ((step>5)&&((rrr(step) >= 0.99 && rrr_d(step) >= 0.99)|| (dN < 1e-5 && rrr_z(step) > 0.6 && rrr_fcd(step) <0.3) )), break, end
        %if ((step>5)&&(rrr_z(step)-rrr_z(step-1)<=-0.10)),break,end
        
        step = step + 1; %counter
        
    end
    %<-----------------------------------------End while loop, End estimation ---------
    
    %--------------------------------------------------------------------------
    %% End estimation, save result
    
    %find the best results
    [rrr_z_max,indx_max] = max(rrr_z);
    [rrr_fcd_min,indx1_min] = min(rrr_fcd(rrr_fcd~=0));
    [rrr_total_min,indx_min] = min(rrr_total(rrr_total~=0));
    Para_E = Para_E_step_save(:,indx_min);
    rrr_fc = rrr_z(indx_min);
    rrr_fcdks = rrr_fcd(indx_min);
    meta = metastable_step(indx_min);
    syn = synchrony_step(indx_min);
    
    disp(Para_E)
    disp(['FC_r=' num2str(rrr_fc)])
    disp(['FCD_r=' num2str(rrr_fcdks)])
    disp(['total_r=' num2str(rrr_total_min)])
    disp(indx_min)
    saved_date = fix(clock);
    %save estclear all
    
    save( [save_dir '/Estimated_Parameter_6p2_' num2str(run_time),'_data_adni_low_CN'],'meta','syn','Para_E','rrr_z_max','FC','rrr_total_min','rrr_fcd_min','rrr_fc','rrr_fcdks','Myelin','atrophy','Para_E_step_save','rrr_total','rrr_fcd','rrr_z');
    %save( [save_dir '/Estimated_Parameter_4p_' list(sub).name],'Para_E','rrr_z_max');
    save('save_all_for_check_62parameter')
    poolobj = gcp('nocreate');
    delete(poolobj);
end
% end
rmpath(lib_dir);
disp('finish');
