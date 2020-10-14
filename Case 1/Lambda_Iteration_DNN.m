function [p_hat,p,delt] = Lambda_Iteration_DNN(Pd)
% clc
% clear
% Pd = 100;
Lambda_L =0;
Lambda_H =4e3;
Lambda_max = Lambda_H;
epsilon = 1e-3;
T = 28;
alpha = 2;
N = 3;
a = [0.0025,0.004,0.005];
b = [1,1.6,1.25];
c = [170,170,200];
up = [100,100,120];
low = [10,20,10];
Q = (3*Lambda_max*alpha*N)/(4*0.0025);
n = floor(-2+log2((Lambda_max*(Q)^(T-2))/(0.0025*epsilon)));
p_hat = zeros(N,T+1);
p = zeros(N,T+1);
delt = zeros(1,T+1);
%------------------------------------------------------
[Lambda_H,Lambda_L,p_hat,p,delt] = t0(Lambda_H,Lambda_L,Pd,alpha,N,a,b,up,low,p_hat,p,delt);
for j = 1:27
[Lambda_H,Lambda_L,p_hat,p,delt] = t1_27(Lambda_H,Lambda_L,Pd,alpha,N,a,b,up,low,p_hat,p,j,n,delt);
end
[p_hat,p] = t28(Lambda_H,Lambda_L,N,a,b,up,low,p_hat,p,28);




function [Lambda_H,Lambda_L,p_hat,p,delt] = t0(Lambda_H,Lambda_L,Pd,alpha,N,a,b,up,low,p_hat,p,delt)
Lambda_M = (Lambda_H+Lambda_L)/2;
for i = 1:N
    p_hat(i,1) = (Lambda_M-b(i))/(2*a(i));
    if p_hat(i,1)> up(i)
        p(i,1) = up(i);
       
    elseif p_hat(i,1)< low(i)
        p(i,1) = low(i);
        
    else
        p(i,1) = p_hat(i,1);
       
    end
end
delt(1) = sum(p(:,1))-Pd;
K=1/(1+exp(-alpha*delt(1)));
Lambda_H = K*Lambda_M + (1-K)*Lambda_H;
Lambda_L = (1-K)*Lambda_M + K*Lambda_L;
end



function [Lambda_newH,Lambda_newL,p_hat,p,delt] = t1_27(Lambda_H,Lambda_L,Pd,alpha,N,a,b,up,low,p_hat,p,tindex,n,delt)
Lambda_M = (Lambda_H+Lambda_L)/2;
for i = 1:N
    p_hat(i,tindex+1) = (Lambda_M-b(i))/(2*a(i));
    if p_hat(i,tindex+1)> up(i)
        p(i,tindex+1) = up(i);
       
    elseif p_hat(i,tindex+1)< low(i)
        p(i,tindex+1) = low(i);
        
    else
        p(i,tindex+1) = p_hat(i,tindex+1);
       
    end
end
delt(tindex+1) = sum(p(:,tindex+1))-Pd;
K=1/(1+exp(-alpha*delt(tindex+1)));
Lambda_newH = Lambda_H-(BinaryApproximateMultiplication(K,Lambda_H,n)/2)+(BinaryApproximateMultiplication(K,Lambda_L,n))/2;
Lambda_newL = Lambda_H/2-(BinaryApproximateMultiplication(K,Lambda_H,n)/2)+ Lambda_L/2+(BinaryApproximateMultiplication(K,Lambda_L,n))/2;
end

function [p_hat,p] = t28(Lambda_H,Lambda_L,N,a,b,up,low,p_hat,p,tindex)
Lambda_M = (Lambda_H+Lambda_L)/2;
for i = 1:N
    p_hat(i,tindex+1) = (Lambda_M-b(i))/(2*a(i));
    if p_hat(i,tindex+1)> up(i)
        p(i,tindex+1) = up(i);
       
    elseif p_hat(i,tindex+1)< low(i)
        p(i,tindex+1) = low(i);
        
    else
        p(i,tindex+1) = p_hat(i,tindex+1);
       
    end
end
end

function [out] = BinaryApproximateMultiplication(x,y,n)
m = ceil(log2(x));
x_Binary = BinaryExpansion(x,n);
x_Binary_sum = 0;
for i = 1:m+n+1    
    x_Binary_sum = x_Binary_sum + x_Binary(1,i) * 2^(m-i+1);
end
out = x_Binary_sum * y;
end

function [x_Binary] = BinaryExpansion(x,n)
m = ceil(log2(x));
xi = zeros(1,m+n+1);
xi(1,1) = x;
x_Binary = zeros(1,m+n+1);
for i = 1:m+n+1
if xi(1,i) >= 2^(m-i+1)
    x_Binary(1,i) = 1;
    if i<m+n+1
        xi(1,i+1) = xi(1,i) - 2^(m-i+1);
    end
else
   x_Binary(1,i) = 0; 
   if i<m+n+1
        xi(1,i+1) = xi(1,i);
   end 
end
end
end


end
