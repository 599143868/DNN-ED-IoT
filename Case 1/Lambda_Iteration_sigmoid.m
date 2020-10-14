function [Phat,PP,P_error] = Lambda_Iteration_sigmoid(Pd)

% clc
% clear

% Initial Value
Lambda_L =0;
Lambda_H =4e3;

% Algorithm Stop condition
error = 1e-5;

%Parameters of TGs and demand
% Pd = 100;
alpha = 2;

alpha_1 = 0.0025; 
beta_1 = 1;
gamma_1 = 170;
P1_Up = 100;
P1_Low = 10;

alpha_2 = 0.004;
beta_2 = 1.6;
gamma_2 = 170;
P2_Up = 100;
P2_Low = 20;

alpha_3 = 0.005;
beta_3 = 1.25;
gamma_3 = 200;
P3_Up = 120;
P3_Low = 10;

%iteration step
k = 0;


while(1)
    
    k = k + 1;
    
    Lambda_M = (Lambda_H + Lambda_L)/2;
    
    %Calculate P1
    P1 = (Lambda_M -beta_1)/(2*alpha_1);
    Phat(1,k)=P1;
    if (P1 > P1_Up)
        P1 = P1_Up;
    end
    
    if (P1 < P1_Low)
        P1 = P1_Low;
    end
    PP(1,k)=P1;
    %Calculate P2
    P2 = (Lambda_M -beta_2)/(2*alpha_2);
    Phat(2,k)=P2;
    if (P2 > P2_Up)
        P2 = P2_Up;
    end
    
    if (P2 < P2_Low)
        P2 = P2_Low;
    end
    PP(2,k)=P2;
    %Calculate P3
    P3 = (Lambda_M -beta_3)/(2*alpha_3);
    Phat(3,k)=P3;
    if (P3 > P3_Up)
        P3 = P3_Up;
    end
    
    if (P3 < P3_Low)
        P3 = P3_Low;
    end
    PP(3,k)=P3;  
    
    %Iteration
    P_total = P1 + P2 + P3;
    
    P_error(k) = P_total - Pd;
    
    K=1/(1+exp(-alpha*P_error(k)));
    Lambda_H = K*Lambda_M + (1-K)*Lambda_H;
    Lambda_L = (1-K)*Lambda_M + K*Lambda_L;
    
    
    
%     if(P_error > 0)
%         Lambda_H = Lambda_M;
%     else
%         Lambda_L = Lambda_M;
%     end
    
    
    if (abs(Lambda_H - Lambda_L)<error)
        break;
    end
    
    Z(k,1) = P1;
    Z(k,2) = P2;
    Z(k,3) = P3;
    
end

% P = [P1 P2 P3]
% 
% FDiscost = alpha_1*P1^2 + beta_1*P1 + gamma_1 +...
%         alpha_2*P2^2 + beta_2*P2 + gamma_2 +...
%         alpha_3*P3^2 + beta_3*P3 + gamma_3;
%     
% FCencost = alpha_1*30^2 + beta_1*30 + gamma_1 +...
%         alpha_2*19.2688^2 + beta_2*19.2688 + gamma_2 +...
%         alpha_3*50.7312^2 + beta_3*50.7312 + gamma_3
end
% figure(1)
% stairs(Z(:,1),'r-','LineWidth',3);
% hold on;
% stairs(Z(:,2),'k-','LineWidth',3);
% hold on;
% stairs(Z(:,3),'g-','LineWidth',3);
% legend('DG_1','DG_2','DG_3');
% xlabel('\fontsize{14}Iterations');ylabel('\fontsize{14}Estimates Output');
% axis([1 14 0 100]);
% grid on;

% end