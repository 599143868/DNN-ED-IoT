function P = Finetune(P,Pd)
up = [200,80,50,35,30,40];
low = [50,20,15,10,10,12];
    for i = 1:length(P)
        if P(i) > up(i)
            P(i) = up(i);
        elseif P(i) < low(i)
            P(i) = low(i);
        end
    end  
while(1)
    delta = Pd-sum(P);
    if delta < 1e-4
        break;
    end
    delta_ave = delta/length(P);
    for i = 1:length(P)
        P(i) = P(i) + delta_ave;
        if P(i) > up(i)
            P(i) = up(i);
        elseif P(i) < low(i)
            P(i) = low(i);
        end
    end   
end    
end