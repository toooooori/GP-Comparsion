function yt_s=GP_Multivariate_solCGP(hyp_sol, X, y, yt, dj)



[K_yy, Kd_yystar, Kdd_ystar, Kd_ystary]=covSEard_GP(hyp_sol.cov, y, yt, dj);

%--------Get the maximum points--------------------------------------------

L_yy=jitterchol(K_yy);
X_mu=X-hyp_sol.mean;

alpha_mu=L_yy'\(L_yy\X_mu);

mu_Kdd=Kd_ystary*alpha_mu;

v_mu=L_yy\Kd_yystar;

cov_Kdd=Kdd_ystar-v_mu'*v_mu;

nstar=size(yt,1);

%---------------------Initial Start Conditions-----------------------------

set_num=0;
prob_y=normcdf(zeros(nstar-set_num,1),mu_Kdd,diag(sqrt(cov_Kdd)));



th_hd=1e-200;

yt_s=[];
yt_starg=yt;
a=[];

while max(prob_y)>th_hd
    
    set_num=set_num+1; 
    
    temp=find(prob_y==max(prob_y));
    lg=length(temp);
    if lg~=1
      a_mat=temp(1);
      temp_idx=0;
     for i=1:lg
        if temp(i)~=(a_mat+i-1) 
           temp_idx=i-1;
        break;
        end
     end    
     if temp_idx~=0;
       if mod(temp_idx,2)==0
          temp_va=yt(temp(temp_idx/2),:);
       else
         temp_va=yt(temp((temp_idx+1)/2),:);
       end
     else
       if mod(lg,2)==0
          temp_va=yt(temp(lg/2),:);
       else
          temp_va=yt(temp((lg+1)/2),:);
       end
     end
    else
       temp_va=yt(temp,:);
    end
    
    yt_s=[yt_s;temp_va];
    
   %---------------Compute Mean and Covariance of yt_s locations----------- 
   
%     Kd_ytsy=covSEard_G(hyp_sol.cov,yt_s,y,1,2);
%     Kd_yyts=covSEard_G(hyp_sol.cov,y,yt_s,1,2);
%     Kdd_yts=covSEard_G(hyp_sol.cov, yt_s, yt_s,2,2); 


    [K_yy, Kd_yyts, Kdd_yts, Kd_ytsy]=covSEard_GP(hyp_sol.cov, y, yt_s, dj);

    
    mu_Kddnew=Kd_ytsy*alpha_mu;

    v_munew=L_yy\Kd_yyts;
    cov_Kddnew=Kdd_yts-v_munew'*v_munew;

    
   %--------------Draw Zd values from its marginal distribution------------

   
   M=501;  
   
   Zd=zeros(set_num,M);

   jitter=1e-6;

   for k=2:M
 
   %--------------Gibbs sampling procedure-----------------------------------
    mu_S=zeros(set_num,1);
    sigma_S=zeros(set_num,1);

    for i=1:set_num

      %------------obtain the conditional proposal distribution------------------
       if set_num~=1
          temp_s=cov_Kddnew(i,:);
          temp_s(i)=[];
          temp_S=cov_Kddnew;
          temp_S(i,:)=[];
          temp_S(:,i)=[];
          
          if i>1 && i<set_num
             temp_zs=[Zd((1:i-1),k);Zd((i+1:set_num),k-1)];
          elseif i==1
             temp_zs=Zd(2:set_num,k-1);
          else
             temp_zs=Zd(1:set_num-1,k);
          end
          
          [ntemp, mtemp]=size(temp_S);
          
          mu_dSi=mu_Kddnew;
          mu_dSi(i)=[];
          
          tempS_chol=jitterchol(temp_S);
          temp_alpha=tempS_chol'\(tempS_chol\(temp_zs-mu_dSi));
          mu_S(i)=mu_Kddnew(i)+temp_s*temp_alpha;
          
          tempS_v=tempS_chol\(temp_s');
          sigma_S(i)=sqrt(cov_Kddnew(i,i)-tempS_v'*tempS_v);
          
         
       else
          mu_S(i)=mu_Kddnew;
          sigma_S(i)=sqrt(cov_Kddnew);
       end
       
    %---------------Gibbs Sampling Procedure One--------------------
    %---------------Truncated Normal Distribution-------------------

     
       temprand=normcdf(0,mu_S(i),sigma_S(i));
       u=temprand+(1-temprand)*rand(1);              % GP values is non-decreasing w.r.t 2nd input
       Zd(i,k)=norminv(u,mu_S(i),sigma_S(i));
   
       if isnan(Zd)
          b=1
       end
       if Zd(i,k)==Inf || Zd(i,k)==-Inf
          Zd(i,k)=0;
          a=[a;i,k]
       end
        
    end  
    
   end


 if nstar-set_num>0
   Zd_temp=mean(Zd(:,(51:M)),2);
   yt_sl=yt_starg(~ismember(yt_starg,yt_s,'rows')',:);                 % left inputs

   %---------------Compute Mean and Covariance-----------------------------
   
   Kcov=[Kdd_yts,  Kd_ytsy; Kd_yyts, K_yy];
   Kcov_chol=jitterchol(Kcov);
   
   Kmean=[Zd_temp; X-hyp_sol.mean];
   alpha_Kcov=Kcov_chol'\(Kcov_chol\Kmean);
   
   Kdd_ytslys=covSEard_GPD(hyp_sol.cov,yt_sl,yt_s,dj);
   Kdd_ysytsl=covSEard_GPD(hyp_sol.cov,yt_s,yt_sl,dj);
   
   
   [K_yy,Kd_yytsl, Kdd_ytsl,Kd_ytsly]=covSEard_GP(hyp_sol.cov,y,yt_sl,dj);
   temp_list=[Kdd_ytslys, Kd_ytsly];
   temp_up=[Kdd_ysytsl; Kd_yytsl];
   mu_Kl=temp_list*alpha_Kcov;
   
   v_Kl=Kcov_chol\(temp_up);
 
   
   
   cov_Kl=Kdd_ytsl-v_Kl'*v_Kl;
   cov_Kl=cov_Kl/(M-50);
   
   prob_y=normcdf(zeros(nstar-set_num,1),mu_Kl,diag(sqrt(cov_Kl)));
   
   yt=yt_sl;
   
%    figure;
%    plot(yt_sl,prob_y,'.')

 else
   yt_s=yt_starg;
 break;
 end

end


