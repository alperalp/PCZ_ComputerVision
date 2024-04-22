% Read the first image
lena = imread('lena.png');

% Read the second image
mountains = imread('mountains.jpg')
% Show two images in one window
figure
subplot(1,2,1), imshow(lena)
subplot(1,2,2), imshow(mountains)

% Show fused image
figure
imshowpair(lena,mountains)