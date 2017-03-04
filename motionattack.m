%% Motion Blur
function blurredimage = motionattack(watermarked_image)
H = fspecial('motion',3,3);
blurredimage = imfilter(watermarked_image,H,'replicate');
end