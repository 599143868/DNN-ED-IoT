% clc
% clear
function [PP,A_lpha,B_eta] = Lambda_Iteration(Pd,u_c)
% Initial Value
Lambda_L =0;
Lambda_H =5e3;

% Algorithm Stop condition
error = 1e-5;

%Parameters of TGs and demand
% Pd = 20;
% u_c = [0 0 0 0 0 1];

NumAgent = sum(u_c);

A_lpha = [0.00375 0.0175 0.0625 0.00834 0.025 0.025];
B_eta = [2 1.75 1.0 3.25 3.0 3.0];
P_max = [200 80 50 35 30 40];
P_min = [50 20 15 10 10 12];

Alpha = A_lpha.*u_c;
Beta = B_eta.*u_c;
Pmax = P_max.*u_c;
Pmin = P_min.*u_c;
Alpha(Alpha==0)=[];
Beta(Beta==0)=[];
Pmax(Pmax==0)=[];
Pmin(Pmin==0)=[];

%iteration step
k = 0;

while(1)

    k = k + 1;

    Lambda_M = (Lambda_H + Lambda_L)/2;
 for i = 1:NumAgent   
    %Calculate P
    P(i) = (Lambda_M -Beta(i))/(2*Alpha(i));
    if (P(i) > Pmax(i))
        P(i) = Pmax(i);
    end
    
    if (P(i) < Pmin(i))
       P(i) = Pmin(i);
    end
 end
       
    %Iteration
    P_total = sum(P);
    
    P_error = P_total - Pd;
    
    if(P_error > 0)
        Lambda_H = Lambda_M;
    else
        Lambda_L = Lambda_M;
    end
    
    
    if (abs(Lambda_H - Lambda_L)<error)
        break;
    end

end

L = 1;
for i = 1:6
    if(u_c(i)==1)
        PP(i) = P(L);
        L = L+1;
    else
        PP(i) = 0;
    end
end
end