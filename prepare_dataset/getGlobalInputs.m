function count = getGlobalInputs(strPath, save_path, nums)
    % strPath: the root directory the original image
    % save_path: the save directory
    % nums: the number of local inputs you want, nums also can be set to random
    count = 0;
    path=strPath;
    Files = dir(path);
    LengthFiles = length(Files);
    for iCount = 1:LengthFiles       
        name = Files(iCount).name;    
        
        if strcmp(name, '.') || strcmp(name, '..')
           continue; 
        end
        if Files(iCount).isdir
             s = [path  name '\']; 
             disp(['正在生成', s]);
             count = count + getGlobalInputs(s, save_path, nums);
        else         
            filename = [path, '\', Files(iCount).name];
            fprintf('%s\n', filename);            
            ImageKMeans(filename, nums, save_path,Files(iCount).name);
            count = count + 1;
        end       
    end
end

