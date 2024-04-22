% EXERCISE 01

% Read the image
image = imread('lena.png');
figure,imshow(image)

% Convert to grayscale
grayscale = rgb2gray(image)
figure,imshow(grayscale)

% Save the grayscale image
imwrite(grayscale,'lena_gray.png')

% EXERCISE 02

% Read the first image
lena = imread('lena.png');

% Read the second image
cat = imread('cat.png');
% Show two images in one window
figure
subplot(2,2,1), imshow(lena)
subplot(2,2,2), imshow(cat)
subplot(2,2,3), imshowpair(lena,cat)

% Show fused image



% EXERCISE 03

% Source path
src = 'Pom1';
src_files = dir([src '/*.png']);
% Destination path
dst = 'Pom2';

for i=1:length(src_files)
    filename = [src '/' src_files(i).name];
    image = imread(filename);
    grayscale = rgb2gray(image);
    dst_file = [dst '/' src_files(i).name];
    imwrite(grayscale,dst_file);
    figure,imshow(grayscale)
end

% EXERCISE 04
src = 'Pom1';
src_files = dir([src, '/*.png']);

dst = 'Pom3';

for i=1:length(src_files)
    filename = [src '/' src_files(i).name];
    image = imread(filename);
    image_resized = imresize3(image,[227 227 3]);
    dst_file = [dst '/' src_files(i).name];
    imwrite(image_resized,dst_file);
    figure,imshow(image_resized);
end
