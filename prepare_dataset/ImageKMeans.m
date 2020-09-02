function  ImageKMeans(filename, numCluster, save_path, name)
% filename: the full image path
% name: the name of  the image
% sort_CenterColor: the rgb global input
% gray: the global input mask
% color_index: the index map to generate the color map
image = double(imread(filename))/255;
imageSize = size(image);

if numel(imageSize)<=2
    return; 
end

image = reshape(image, [imageSize(1)*imageSize(2), 3]);
opts = statset('Display','off','MaxIter',200);
[Index, CenterColor] = kmeans(image, numCluster, 'Options', opts);
Index=reshape(Index, imageSize(1), imageSize(2));
CenterColor = reshape(CenterColor, 1, numCluster, 3);
sort_CenterColor = zeros(1, numCluster, 3);

% Sort color themes by frequency
order = cell(numCluster, 1);
count = zeros(numCluster, 1);
for i = 1 : numCluster
    order{i, 1} = find(Index == i);
    count(i, 1) = length(order{i, 1});
end

for i = 1 : numCluster
    [~, max_index] = max(count);
    sort_CenterColor(1, i, :) = CenterColor(1, max_index, :);
    Index(order{max_index}) = i;
    count(max_index) = -1;
end
% save
gray = rgb2gray(sort_CenterColor);
sort_CenterColor = uint8(sort_CenterColor * 255);
gray = uint8(gray);
color_index = uint8(Index);
if ~exist([save_path, 'color_theme'], 'dir')
    mkdir([save_path, 'color_theme']);
    mkdir([save_path, 'color_theme_mask']);
    mkdir([save_path, 'index_map']);
end
name=split(name,'.jpg');
name=char(name(1));
imwrite(sort_CenterColor, [save_path, 'color_theme','\', name, '.png']);
imwrite(gray, [save_path, 'color_theme_mask', '\',name, '.png']);
imwrite(color_index, [save_path,'index_map', '\',name, '.png']);
end

