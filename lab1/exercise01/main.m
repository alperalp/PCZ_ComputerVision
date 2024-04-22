% Read the image
image = imread('lena.png');
figure,imshow(image)

% Convert to grayscale
grayscale = rgb2gray(image)
figure,imshow(grayscale)

% Save the grayscale image
imwrite(grayscale,'lena_gray.png')