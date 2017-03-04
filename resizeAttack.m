%% Resize Image
function resizedimage = resizeAttack(watermarked_image)
resizedimage = imresize(watermarked_image, [128 128]);
resizedimage = imresize(resizedimage, [256 256]);
end