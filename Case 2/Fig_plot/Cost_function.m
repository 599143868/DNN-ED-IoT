function Cost = Cost_function(P)
Alpha = [0.00375 0.0175 0.0625 0.00834 0.025 0.025];
Beta = [2 1.75 1.0 3.25 3.0 3.0];

Cost = 0;
for i = 1:length(P)
    Cost = Cost + Alpha(i)*P(i)^2 + Beta(i)*P(i);
end
end