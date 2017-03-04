function [ out ] = PSNR( pic1,pic2 )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
e=MSE(pic1,pic2);
%m=max(max(pic1));
%out = m;
out=10*log10((255^2)/e);
end

