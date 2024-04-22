src = 'Pom3';
src_files = dir([src, '/*.png']);

for i=1:length(src_files)
    filename = [src '/' src_files(i).name];
    image = imread(filename);
    image_resized = imresize3(image,[227 227 3]);
    imwrite(image_resized,filename);
    figure,imshow(image_resized);
end