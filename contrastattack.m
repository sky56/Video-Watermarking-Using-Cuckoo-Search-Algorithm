%% Contrast Image
function constrastimage = contrastattack(watermarked_image)
constrastimage = imadjust(watermarked_image, [0 0.8], [0 1]);
end