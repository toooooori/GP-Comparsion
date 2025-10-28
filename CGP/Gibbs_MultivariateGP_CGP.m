function Zval=Gibbs_MultivariateGP_CGP(hyp_sol, X, y, yt, yt_s, dj)

n=size(y,1);
nstar=size(yt,1);
m=size(yt_s,1);


%----------------Calculate mean and covariance for Z'(y_s)-----------------

[K_yy, Kd_yys, Kdd_ys, Kd_ysy]=covSEard_GP(hyp_sol.cov, y, yt_s,dj);


L_ys=jitterchol(Kdd_ys);


% inv_covX=L_yy'\(L_yy\eye(size(K_yy,1)));

mean_dX=(L_ys'\(L_ys\Kd_ysy))';                                                 % A matrix
cov_dX=K_yy-mean_dX*Kd_ysy;                                               % B matrix                                                                               

temp_covdX=jitterchol(cov_dX);
% inv_dX=temp_covdX'/(temp_covdX/eye(size(cov_dX, 1)));

%----------------Calculate mean and covariance for Z(y^*)------------------

[K_ytyt, Kd_ytys, Kdd_ys, Kd_ysyt]=covSEard_GP(hyp_sol.cov, yt, yt_s,dj);

K_yyt=covSEardS(hyp_sol.cov,y,yt);


Sigma_11=[K_yy, K_yyt; K_yyt', K_ytyt] ;
Sigma_12=[Kd_yys; Kd_ytys];
Sigma_21=[Kd_ysy, Kd_ysyt];

v_ys=L_ys\Sigma_21;


Lambda=Sigma_11-v_ys'*v_ys;


Lambda_11=Lambda(1:n,1:n);
Lambda_12=Lambda(1:n, (n+1):(n+nstar));
Lambda_21=Lambda((n+1):(n+nstar), 1:n);
Lambda_22=Lambda((n+1):(n+nstar), (n+1):(n+nstar));



L_Lb=jitterchol(Lambda_11);
v_Lb=L_Lb\Lambda_12;

Lambda_sig=Lambda_22-v_Lb'*v_Lb;


K3=Kdd_ys;


%----------------Initial Conditions for Gibbs Sampling---------------------

M=1000;
Z=zeros(nstar,M);                                                            % indicate Z(s)
Zd=zeros(m,M);                                                               % indicate Z'(s)
Zd_p=zeros(m,M);                                                             % indicate Z'^{+}(s) 


%----------------------Gibbs Sampling Procedure----------------------------

for k=2:M
    

mu_m=zeros(m,1);
nu_m=zeros(m,1);

theta_m=zeros(m,1);
delta_m=zeros(m,1);

%--------------Gibbs sampling procedure-----------------------------------
for i=1:m

%------------obtain the conditional proposal distribution------------------


%------------unconstraint posterior proposal distribution------------------
    
    temp_s=K3(i,:);
    temp_s(i)=[];
    temp_S=K3;
    temp_S(i,:)=[];
    temp_S(:,i)=[];
    
%------------unconstraint prior proposal distribution----------------------
    
    temp_a=mean_dX(:,i);
    if i==1
       temp_A=mean_dX(:, (i+1:m));
    elseif i==m
       temp_A=mean_dX(:,(1:i-1));
    else
       temp_A=[mean_dX(:,(1:i-1)), mean_dX(:, (i+1:m))];
    end


    
    
    if i>1 && i<m
        temp_zp=[Zd((1:i-1),k);Zd((i+1:m),k-1)];
        temp_zpp=[Zd_p((1:i-1),k);Zd_p((i+1:m),k-1)];
    elseif i==1
        temp_zp=Zd(2:m,k-1);
        temp_zpp=Zd_p(2:m,k-1);
    else
        temp_zp=Zd(1:m-1,k);
        temp_zpp=Zd_p(1:m-1,k);
    end
    
%--------------normal distribution for prior---------------------  

    tempS_chol=jitterchol(temp_S);
     
    v_tempS=tempS_chol\(temp_s');
    
    alpha_tempS=tempS_chol'\(tempS_chol\temp_zp);
 
    
    mu_m(i)=temp_s*alpha_tempS;
    nu_m(i)=K3(i,i)-v_tempS'*v_tempS;
    nu_sqrt=sqrt(nu_m(i));
    
%----------unconstraint posterior normal distribution------------ 

    v_dX=temp_covdX\temp_a;  
       
    temp_th1=(X-hyp_sol.mean)-temp_A*temp_zpp;
    
    alpha_dX=temp_covdX'\(temp_covdX\temp_th1);

    temp_th=temp_a'*alpha_dX+mu_m(i)*nu_m(i)^(-1);
    temp_dt1=nu_m(i)^(-1)+v_dX'*v_dX;
    temp_dt=1/temp_dt1;
    theta_m(i)=temp_dt*temp_th;
    delta_m(i)=sqrt(temp_dt);

%---------------Gibbs Sampling Procedure One--------------------

%---------------Binomial Selection Algorithm-------------------
  
    eps=0;    
    temp_ks=normcdf(eps,mu_m(i),nu_sqrt);

    temp_qx=1-normcdf(eps,theta_m(i),delta_m(i));    
    temp_const=temp_qx*sqrt(delta_m(i)/nu_sqrt)*exp(-mu_m(i)^2/(2*nu_m(i))+theta_m(i)^2/(2*temp_dt));

    q_const=temp_const/(temp_ks+temp_const);
    
     if temp_const==Inf   
       q_const=1;
    end
    
       
    if isnan(temp_const)
       warning('NaN value')
       break;
    end
    
    if temp_ks==0 && temp_const==0
       temp_ks=1e-6;
        q_const=temp_const/(temp_ks+temp_const);
    end
   
    biv_num=binornd(1,q_const);
    
    if biv_num==1
       u=temp_ks*rand(1);
       Zd(i,k)=norminv(u,mu_m(i),nu_sqrt);
       Zd_p(i,k)=0;
    else
       u=1-(temp_qx-temp_qx*rand(1));
       Zd(i,k)=norminv(u,theta_m(i),delta_m(i));
       Zd_p(i,k)=Zd(i,k);
    end
  
   
end
mu_lam=hyp_sol.mean*ones(n+nstar,1);

alpha_K3=L_ys'\(L_ys\Zd_p(:,k));

MU=mu_lam+Sigma_12*alpha_K3;
mu_z=MU((n+1):(n+nstar));
mu_x=MU(1:n);

%---------------Gibbs Sampling Procedure Two--------------------
%---------------Multivariate Normal Distribution----------------

Z_mu=mu_z+Lambda_21*(L_Lb'\(L_Lb\(X-mu_x)));
Z_sigma=Lambda_sig;
Z(:,k)=Z_mu+jitterchol(Z_sigma)*randn(nstar,1);

end

%--------------Summarize Gibbs Results-------------------------------------

Zval=Z(:,501:1000);  
Z_qt=quantile(Zval', [0.025,0.975]);  
Z_qt1=Z_qt(1,:);
Z_qt2=Z_qt(2,:);

