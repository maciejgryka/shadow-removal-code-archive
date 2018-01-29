function [data, query] = getfield()
    PATCH_SIZE = 3;

    img_dir = 'C:\Work\research\shadow_removal\experiments\test_images_real\small\';
    shad_path = [img_dir, 'real157_shad.png'];
    mask_path = [img_dir, 'real157_smask.png'];

    shad = im2double(imread(shad_path));
    colors = shad ./ repmat(sum(shad, 3), [1, 1, 3]);
    colors = colors / max(max(max(colors)));
    colors = cat(3, colors(:,:,2) ./ colors(:,:,1), colors(:,:,3) ./ colors(:,:,1));
    gradients = shad(:,:,1) ./ imfilter(shad(:,:,1), fspecial('gaussian', 3, 1));
    img = cat(3, gradients, colors);
    mask = imread(mask_path);
    mask = im2double(mask(:,:,1));

    % get coordinates of all pixels outside of the shadow region
    [data, data_coords] = get_patch_matrix(img, 1.0 - mask, PATCH_SIZE);
    [query, query_coords]  = get_patch_matrix(img, mask, PATCH_SIZE);
    size(data)
    size(query)
%
% %     % subsample the query matrix
% %     rows = randperm(size(query, 1));
% %     N_SHOW = 2000;
% %     query = query(rows(1:N_SHOW), :);
% %     query_coords = query_coords(rows(1:N_SHOW), :);
%
    tic;
    inds = knnsearch(data, query, 'k', 5, 'NSMethod', 'kdtree');
    toc;
%
% %     imshow(shad);
% %     hold on;
%
%     for p = 1:length(inds)
% %         plot([query_coords(p, 1), data_coords(inds(p), 1)],...
% %              [query_coords(p, 2), data_coords(inds(p), 2)], '-r');
%         shad(query_coords(p, 2), query_coords(p, 1), :) = shad(data_coords(inds(p), 2), data_coords(inds(p), 1), :);
%     end
%     imshow(shad);
%     hold off;
end


function [data, coords] = get_patch_matrix(img, mask, PATCH_SIZE)
% Return a matrix where each row is a representation of a patch from img.
% One patch is computed around each pixel specified by coords.
    min_x = floor(PATCH_SIZE/2);
    min_y = min_x;
    max_x = size(img,2) - min_x;
    max_y = size(img,1) - min_y;

    % modify the mask to exclude things close to the boundary
    mask(1:min_y, :) = 0.0;
    mask(max_y+1:end, :) = 0.0;
    mask(:, 1:min_x) = 0.0;
    mask(:, max_x+1:end)= 0.0;

    [Y, X] = ind2sub(size(mask), find(mask == 1.0));
    data_dims = get_data_dims(PATCH_SIZE);
    data = zeros(length(Y), data_dims);
    coords = zeros(length(Y), 2);
    for p = 1:length(Y)
        coords(p, :) = [X(p), Y(p)];
        patch = get_patch(img, coords(p, :), PATCH_SIZE);
        data(p, :) = patch(:);
    end
end


function patch = get_patch(img, coords, patch_size)
    x = coords(1) - floor(patch_size/2);
    y = coords(2) - floor(patch_size/2);

    patch = [];
    for d = 1:size(img,3)
        patch = horzcat(patch, img(y:y+patch_size-1, x:x+patch_size-1));
    end
end


function data_dims = get_data_dims(patch_size)
    data_dims = patch_size * patch_size * 3;
end