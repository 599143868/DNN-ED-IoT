clear 
clc

load DNN_output.mat;
load('yourpath/Dataset.mat');
Pd = Dataset(:,1);
Lambda_output = Dataset(:,2:7);

up = [200,80,50,35,30,40];
low = [50,20,15,10,10,12];
Alpha = [0.00375 0.0175 0.0625 0.00834 0.025 0.025];
Beta = [2 1.75 1.0 3.25 3.0 3.0];


for i = 1:length(Dataset)
    DNN_output(i,:) = Finetune(name(i,:),Dataset(i,1));
    DNN_cost(i) = Cost_function(DNN_output(i,:));
    Lambda_cost(i) = Cost_function(Lambda_output(i,:));
end


figure(1)
subplot(3,4,[1,2,3,5,6,7]);plot(Pd,DNN_cost,'k-',Pd,Lambda_cost,'k--','Linewidth',3);xlabel('Total Demand');ylabel('Cost ($/h)');title('Total Cost');legend('DNN','¦Ë-ITE')
legend('location','northwest')
subplot(3,4,4);plot(Pd,DNN_output(:,1),'r-',Pd,Lambda_output(:,1),'r--','Linewidth',3);xlabel('Total Demand');ylabel('Power Output');title('DG 1');legend('DNN','¦Ë-ITE')
legend('location','northwest')
subplot(3,4,8);plot(Pd,DNN_output(:,2),'b-',Pd,Lambda_output(:,2),'b--','Linewidth',3);xlabel('Total Demand');ylabel('Power Output');title('DG 2');legend('DNN','¦Ë-ITE')
legend('location','northwest')
subplot(3,4,9);plot(Pd,DNN_output(:,3),'g-',Pd,Lambda_output(:,3),'g--','Linewidth',3);xlabel('Total Demand');ylabel('Power Output');title('DG 3');legend('DNN','¦Ë-ITE')
legend('location','northwest')
subplot(3,4,10);plot(Pd,DNN_output(:,4),'y-',Pd,Lambda_output(:,4),'y--','Linewidth',3);xlabel('Total Demand');ylabel('Power Output');title('DG 4');legend('DNN','¦Ë-ITE')
legend('location','northwest')
subplot(3,4,11);plot(Pd,DNN_output(:,5),'m-',Pd,Lambda_output(:,5),'m--','Linewidth',3);xlabel('Total Demand');ylabel('Power Output');title('DG 5');legend('DNN','¦Ë-ITE')
legend('location','northwest')
subplot(3,4,12);plot(Pd,DNN_output(:,6),'c-',Pd,Lambda_output(:,6),'c--','Linewidth',3);xlabel('Total Demand');ylabel('Power Output');title('DG 6');legend('DNN','¦Ë-ITE')
legend('location','northwest')
