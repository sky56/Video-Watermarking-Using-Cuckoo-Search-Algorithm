function CorrectedImage = gammacorrection(Image,GammaValue)
Image = double(Image);
CorrectedImage = 255 * (Image/255).^ GammaValue;
end