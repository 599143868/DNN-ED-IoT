clc
clear

for k = 1:151
[Phat,PP,P_error] = Lambda_Iteration_sigmoid(k+49);
[p_hat,p,delt] = Lambda_Iteration_DNN(k+49);
p1(:,k) = PP(:,end);
p2(:,k) = p(:,end);
error(:,k) = PP(:,end)-p(:,end);
end
marker_idx = 1:5:151;
figure(1)
subplot(6,1,1);
plot(50:200,p1(1,:),'b-*','MarkerIndices',marker_idx);hold on;plot(50:200,p2(1,:),'r');legend('Lambda Iteration','DNN');xlabel('Pd');ylabel('P1');
subplot(6,1,2);
plot(50:200,error(1,:),'m');xlabel('Pd');ylabel('Error(P1)');
subplot(6,1,3);
plot(50:200,p1(2,:),'b-*','MarkerIndices',marker_idx);hold on;plot(50:200,p2(2,:),'r');legend('Lambda Iteration','DNN');xlabel('Pd');ylabel('P2');
subplot(6,1,4);
plot(50:200,error(2,:),'m');xlabel('Pd');ylabel('Error(P2)');
subplot(6,1,5);
plot(50:200,p1(3,:),'b-*','MarkerIndices',marker_idx);hold on;plot(50:200,p2(3,:),'r');legend('Lambda Iteration','DNN');xlabel('Pd');ylabel('P3');
subplot(6,1,6);
plot(50:200,error(3,:),'m');xlabel('Pd');ylabel('Error(P3)');





figure(2)
subplot(3,1,1);
plot(0:28,PP(1,:),'b-*',0:28,p(1,:),'r');legend('Lambda Iteration','DNN');xlabel('Steps');ylabel('P1');
subplot(3,1,2);
plot(0:28,PP(2,:),'b-*',0:28,p(2,:),'r');legend('Lambda Iteration','DNN');xlabel('Steps');ylabel('P2');
subplot(3,1,3);
plot(0:28,PP(3,:),'b-*',0:28,p(3,:),'r');legend('Lambda Iteration','DNN');xlabel('Steps');ylabel('P3');





figure(3)
subplot(4,2,7);
plot(0:28,P_error,'b-*',0:28,delt,'r');xlabel('Steps');ylabel('delta');
h1 = refline(0,0);
set(h1,'color','g','LineWidth',1);
legend('Lambda Iteration','DNN-Approximation','0');
subplot(4,2,2);
plot(0:28,Phat(1,:),'b-*',0:28,p_hat(1,:),'r');legend('Lambda Iteration','DNN');xlabel('Steps');ylabel('Phat1');
subplot(4,2,4);
plot(0:28,Phat(2,:),'b-*',0:28,p_hat(2,:),'r');legend('Lambda Iteration','DNN');xlabel('Steps');ylabel('Phat2');
subplot(4,2,6);
plot(0:28,Phat(3,:),'b-*',0:28,p_hat(3,:),'r');legend('Lambda Iteration','DNN');xlabel('Steps');ylabel('Phat3');
