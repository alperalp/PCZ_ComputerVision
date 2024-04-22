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
