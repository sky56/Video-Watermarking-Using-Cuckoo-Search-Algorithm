%% Histogram Equilisation
function histogramimage = histattack(watermarked_image)
histogramimage = adapthisteq(watermarked_image);
end