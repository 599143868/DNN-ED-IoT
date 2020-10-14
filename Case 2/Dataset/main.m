clc
clear;

P_max = [200 80 50 35 30 40];
P_min = [50 20 15 10 10 12];

k = 0;
up = sum(P_max);
low = sum(P_min);
for j = low:0.01:up
        k = k+1;
        Pd(k,:) = j;
        [P(k,:)] = Lambda_Iteration(j,[1,1,1,1,1,1]);
end


Dataset = [Pd P];


