%% sti
clear all
close all

% setup directory for lib, data, cluster, save
main_dir = pwd;

data_dir = fullfile('\6p2_34fc_AD\100');
save_dir = fullfile(main_dir, 'mulproperty\2401\6p2_34fc_AD\100');
list = dir(fullfile(data_dir));
fileNums = length(list);
amplitude=0.02:0.02:1;
lengthFN_1=length(amplitude);
numStil=68;
numNode=68;
TR=3;


% desikan
subnet= [7	4	6	1	5	1	7	5	7	1	5	1	5	7	7	2	4	7	7	1	2	7	2	7	7	6	7	3	2	4	5	5	2	4	7	4	6	1	5	1	7	5	7	1	5	1	5	7	1	2	4	7	4	1	2	7	2	7	7	6	7	3	2	4	5	5	2	4
    ];

for An=3:fileNums
    C2= strsplit(list(An).name,'_');
    Ai = fix(str2double(C2{1})/0.02);
    
    list2 = dir(fullfile([data_dir '\' list(An).name ]));
    fileNums2 = length(list2);
    for Cn=3:fileNums2
        Ci=str2num(list2(Cn).name);
        list3 = dir(fullfile([data_dir '\' list(An).name '\' list2(Cn).name ],'*.mat'));
        fileNums3 = length(list3);
        for Sn=1:fileNums3
            C1 = strsplit(list3(Sn).name,{' ','_','.'});
            Si = str2num(C1{4});
            a=load([data_dir '\' list(An).name '\' list2(Cn).name '\' list3(Sn).name  ]);
            FC_sim=a.FC_sim;
      
            
            FC=FC_sim;
            FC(FC_sim<=0)=0;
            FC(FC_sim>=1)=0;

            W = weight_conversion(FC, 'normalize');
            Clu = clustering_coef_wu(W); %clustring 68*1
            C_real = mean(Clu); 
            
            D=distance_wei(W); 
            [lambda,efficiency,ecc,radius,diameter] = charpath(D,0,1); 
            L_real = lambda;
         
            R = randmio_und(FC,10);
            R(R<=0)=0;
            R(R>=1)=0;
            RW = weight_conversion(R, 'normalize');
            RC = clustering_coef_wu(RW); %clustring 68*1
            C_rand = mean(RC); 
           
            D=distance_wei(RW);  %
            [lambda,efficiency,ecc,radius,diameter] = charpath(D,0,1); 
            L_rand = lambda;
            sw=(C_real/C_rand)/(L_real/L_rand);
            [CI,Q] = modularity_und(FC);%modularity
            Eglob = efficiency_wei(FC);%global efficient
            

            left_GBC = mean2(atanh(FC(1:34,1:34)));
            right_GBC = mean2(atanh(FC(35:end,35:end)));
            inter_GBC =mean2(atanh(FC(35:end,1:34)));
            all_GBC =mean2(atanh(FC));
    
            Meta = a.metastable_sim;
            Syn = a.synchrony_sim;
           
            FCT = mean(FC,2);
            LI_fc=(sum(FCT(1:34))-sum(FCT(34:68)))/(sum(FCT(1:34))+sum(FCT(34:68)));

            if(strcmp(C1{5},'AD'))
                C(Ci,Si) = C_real;
                L(Ci,Si) = L_real;
                SW(Ci,Si) = sw;
                M(Ci,Si) = Q;
                Eg(Ci,Si) = Eglob;
                %                 LF(Ci,Si) = left_fc;
                %                 RF(Ci,Si) = right_fc;
                %                 IntF(Ci,Si) = inter_fc;
                %                 AF(Ci,Si) = all_fc;
                %                 FRsn(Ci,Si,:) = rsn_fc;
                %                 GOF(Ci,Si) = gof;
                %                 SSIM3(Ci,Si) = ssim3;
                %                 %                 LI_D(Ci,Si) = LI_d;
                LI_FC(Ci,Si) = LI_fc;
                META(Ci,Si)=Meta;
                SYN(Ci,Si)=Syn;
                LGBC(Ci,Si) = left_GBC;
                RGBC(Ci,Si) = right_GBC;
                IntGBC(Ci,Si) = inter_GBC;
                GBC(Ci,Si) = all_GBC;
                
            end
            
        end
        disp([list2(Cn).name, 'finish']);
    end
    if exist([save_dir,'\AD'])==0
        mkdir([save_dir,'\AD']);
    end
    save([save_dir,'\AD\','property_' C2{1} '_20.mat'],'META','SYN','C','L','SW','M','Eg');
    save([save_dir,'\AD\','GBC_' C2{1} '_20.mat'],'LGBC','RGBC','IntGBC','GBC');
    save([save_dir,'\AD\','LI_FC_' C2{1} '_20.mat'],'LI_FC');
    disp([C2{1}, 'finish']);
end
disp('finish');






