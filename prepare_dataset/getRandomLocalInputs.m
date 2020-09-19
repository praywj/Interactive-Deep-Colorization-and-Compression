function counter = getRandomLocalInputs(root_dir, save_dir, num)
    % root_dir: the root directory the original image
    % save_path: the save directory
    % num: the number of local inputs you want
    % ori: the rgb image, uint8
    % points_rgb: the rgb local inputs, uint8
    % points_mask: the local inputs mask, uint8
    counter = 0;
    
    file_items = dir(root_dir);
    file_items(1:2) = [];
    item_nums = length(file_items);
    
    for i = 1 : item_nums
        item_name = file_items(i).name;
        item_path = [root_dir, '\', item_name];
        if file_items(i).isdir
            counter = counter + getRandomLocalInputs(item_path, save_dir, num);
        else
            ori = imread(item_path);
            [height, width, C] = size(ori);
            points_mask = zeros(height, width);
            for m = 1 : num
                x = randi([1,height]);
                y = randi([1,width]);
                % Avoid the same local input points
                if points_mask(x, y) ~= 255
                    points_mask(x, y) = 255;
                else
                    m = m-1;
                end
            end
            points_rgb = ori .* points_mask;
            if ~exist([save_dir, '_mask', int2str(num)], 'dir')
               mkdir([save_dir, '_mask', int2str(num)]);
               mkdir([save_dir, int2str(num)]);
            end
            file_name = split(item_name, '.');
            file_name = [char(file_name(1)), '.png'];
            imwrite(points_mask, [save_dir, '_mask', int2str(num), '\', file_name]);
            imwrite(points_rgb, [save_dir, int2str(num), '\', file_name]);
            counter = counter + 1;
            
        end
    end
