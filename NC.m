function [ out ] = NC( pic1,pic2 )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
e=0;
f=0;
g=0;
[m m]=size(pic1);

for i=1:m
    for j=1:m
        e=e+double((pic1(i,j)*pic2(i,j)));
    end
end
for i=1:m
    for j=1:m
        f=f+double((pic1(i,j)^2));
    end
end
for i=1:m
    for j=1:m
        g=g+double((pic2(i,j)^2));
    end
end
out=e/(sqrt(f)*sqrt(g));
end

