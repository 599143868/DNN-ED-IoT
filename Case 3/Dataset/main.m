clc
clear;

P_max = [200 80 50 35 30 40];
P_min = [50 20 15 10 10 12];

k = 0;
for i = 1:2^6-1
    u_c(i,:) = decimal2binary(i);
    Pmax = P_max.*u_c(i,:);
    Pmin = P_min.*u_c(i,:);
    up = sum(Pmax);
    low = sum(Pmin);
    for j = low:0.01:up
        k = k+1;
        Pd(k,:) = j;
%         [P(k,:)] = Lambda_Iteration(j,u_c(i,:));
%         uc(k,:) = i;
    end
end

Dataset = [Pd uc P];
scatter3(Pd,uc,P(:,1),2)

id = randperm(10239);