%% Gaussian Noise Attack
function gaussianNoiseAttack = gaussianNoise(watermarked_image)
gaussianNoiseAttack = imnoise(watermarked_image, 'gaussian', 0, 0.01);
end