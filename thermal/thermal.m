clc;    
close all;  
clear;   
workspace;   
format long g;
format compact;
fontSize = 15;

[img_name,PathName] = uigetfile({'*.jpg;*.jpeg;*.png'},'Select the thermal photo');

 
baseFileName = img_name; 
 
folder = pwd
fullFileName = fullfile(folder, baseFileName);
 
originalRGBImage = imread(fullFileName);
 
subplot(2, 3, 1);
imshow(originalRGBImage, []);
axis on;
caption = sprintf('Original Pseudocolor Image, %s', baseFileName);
title(caption, 'FontSize', fontSize, 'Interpreter', 'None');
drawnow;
grayImage = min(originalRGBImage, [], 3); 

[rows, columns, numberOfColorChannels] = size(originalRGBImage);

colorBarImage = imcrop(originalRGBImage, [1, 20, 17, rows]);
b = colorBarImage(:,:,3);
 
rgbImage = imcrop(originalRGBImage, [81, 1, columns-20, rows]);
 
[rows, columns, numberOfColorChannels] = size(rgbImage);
 
subplot(2, 3, 2);
imshow(rgbImage, []);
axis on;
caption = sprintf('Cropped Pseudocolor Image');
title(caption, 'FontSize', fontSize, 'Interpreter', 'None');
drawnow;
hp = impixelinfo();
 
subplot(2, 3, 3);
imshow(colorBarImage, []);
axis on;
caption = sprintf('Cropped Colorbar Image');
title(caption, 'FontSize', fontSize, 'Interpreter', 'None');
drawnow;
 
set(gcf, 'Units', 'Normalized', 'OuterPosition', [0 0.05 1 0.95]);
 
set(gcf, 'Name', 'Demo', 'NumberTitle', 'Off') 
 
storedColorMap = colorBarImage(:,1,:);
 
storedColorMap = double(squeeze(storedColorMap)) / 255
 
storedColorMap = flipud(storedColorMap);
 
indexedImage = rgb2ind(rgbImage, storedColorMap);
 
subplot(2, 3, 4);
imshow(indexedImage, []);
axis on;
caption = sprintf('Indexed Image (Gray Scale Image)');
title(caption, 'FontSize', fontSize, 'Interpreter', 'None');
drawnow;
 
highTemp = 28.9;
 
lowTemp = 0.9;
 
thermalImage = lowTemp + (highTemp - lowTemp) * mat2gray(indexedImage);
subplot(2, 3, 5);
imshow(thermalImage, []);
axis on;
colorbar;
title('Floating Point Thermal (Temperature) Image', 'FontSize', fontSize, 'Interpreter', 'None');
hp = impixelinfo();  
hp.Units = 'normalized';
hp.Position = [0.45, 0.03, 0.25, 0.05];
 
subplot(2, 3, 6);
histogram(thermalImage, 'Normalization', 'probability');
axis on;
grid on;
caption = sprintf('Histogram of Thermal Temperature Image');
title(caption, 'FontSize', fontSize, 'Interpreter', 'None');
xlabel('Temperature', 'FontSize', fontSize, 'Interpreter', 'None');
ylabel('Frequency', 'FontSize', fontSize, 'Interpreter', 'None');