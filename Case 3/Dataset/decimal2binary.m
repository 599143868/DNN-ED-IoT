function p = decimal2binary(p)
p = dec2bin(p,6);
P = reshape(p,[],1);
clear p;
str2double(P(1));
for i = 1:6
   p(i) = str2double(P(i));
end
end