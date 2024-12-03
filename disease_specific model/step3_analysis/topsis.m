clc
clear all
save_dir='topsis_sti_0-1';
data_dir = '\6p2_34fc\result_0-1_0.02_\';
for A=0.02:0.02:1


load([data_dir 'AD\property_' num2str(A) '_20.mat']);
cn=load('\property_CN.mat');
ad=load('\property_AD.mat');

a=load('\sim_index_AD_1.txt');
index=(a+1)';

distance = csvread('\dis_AD_1.csv',1,1);
[mdis mdisidx] =  min(distance,[],2);
dist = mdis';

X=zeros(length(index),5);

W=[1,1,1,1,1,0.1];


X(:,1) = mean(c(1:end,index),1)';
X(:,2) = mean(l(1:end,index),1)';
X(:,3) = mean(eg(1:end,index),1)';
X(:,4) = mean(Meta(1:end,index),1)';
X(:,5) = mean(Syn(1:end,index),1)';
% X(:,6) = mean(1./dist(1:end,index),1)';

X_CN(:,1) = mean(cn.C,1);
X_CN(:,2) = mean(cn.L,1);
X_CN(:,3) = mean(cn.Eg,1);
X_CN(:,4) = mean(cn.META,1);
X_CN(:,5) = mean(cn.SYN,1);
% X_CN(:,6) = 1./0.74;

X_AD(:,1) = mean(ad.C,1);
X_AD(:,2) = mean(ad.L,1);
X_AD(:,3) = mean(ad.Eg,1);
X_AD(:,4) = mean(ad.META,1);
X_AD(:,5) = mean(ad.SYN,1);
% X_AD(:,6) = 1./2.4248792056282555;
X(end+1,:)=X_CN;
X(end+1,:)=X_AD;

[n,m] = size(X);
% X(:,6) = 1./X(:,6);
Z = X ./ repmat(sum(X.*X) .^ 0.5, n, 1);
Z_CN=Z(end-1,:);
Z_AD=Z(end,:);

D_P = sum([(Z - repmat(max(Z),n,1)) .^ 2 ],2) .^ 0.5;  
D_N = sum([(Z - repmat(min(Z),n,1)) .^ 2 ],2) .^ 0.5;   
% D_P = sum(W.*[(Z - repmat(max(Z_CN),n,1)) .^ 2 ],2) .^ 0.5;   
% D_N = sum(W.*[(Z - repmat(min(Z_AD),n,1)) .^ 2 ],2) .^ 0.5;   
D_P = sum([(Z - repmat(Z_CN,n,1)) .^ 2 ],2) .^ 0.5;   % D+ 
D_N = sum([(Z - repmat(Z_AD,n,1)) .^ 2 ],2) .^ 0.5;   % D- ”Î

% t = Z_CN-Z_AD;
% D_P_r = Z - repmat(Z_CN,n,1);   % D+ 
% D_N_r = Z - repmat(Z_AD,n,1);   % D- 
% 
% for i=1:6
%      D_P_r(find(D_P_r(:,i)<0)) = D_P_r(find(D_P_r(:,i)<0))+t(i)/2;
%      D_N_r(find(D_P_r(:,i)>0)) = D_N_r(find(D_P_r(:,i)>0))+t(i)/2;
% end
% 
% 
% D_P = sum(W.*(D_P_r.^ 2) ,2) .^ 0.5; % D+ 
% D_N =sum( W.*(D_N_r.^ 2) ,2) .^ 0.5; % D- 

S = D_N./ (D_P+D_N);    % 
% S = D_N ./ (D_P+D_N);   
[sorted_S,indexS] = sort(S(1:end-2),'descend');
% stand_S = S_N(1:end-2) / max(S_N(1:end-2));
% [sorted_S,indexS] = sort(stand_S ,'descend');
indexS=index(indexS);

x=1:length(sorted_S);

h1=figure('Units','centimeter','Position',[5 5 10 9.5]);

hold on
plot(x,sorted_S,'-*b','linewidth',2);

xlim([1 16])
box on
set(gca,'linewidth',1.5,'FontSize',15);
saveas(1, ['plot\fcdpdfplot\tosis_0.1_3.png']);
if exist(save_dir)==0
    mkdir(save_dir);
end

saveas(1, [save_dir,'\tosis_' num2str(A) '_2.png']);
save([save_dir,'\tosis_' num2str(A) '_2.mat'],'indexS','sorted_S','X','Z','S');
close(h1)
end
disp('finish');


