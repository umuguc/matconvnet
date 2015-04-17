function [F, G] = deconvnet_pool(X, method, pad, pool, stride)
%DECONVNET_POOL Summary of this function goes here
%   Detailed explanation goes here

X      = padarray(X, [pad([1 3]) 0], 0, 'pre');
X      = padarray(X, [pad([2 4]) 0], 0, 'post');
temp_1 = reshape(1 : numel(X), size(X));

for index_1 = size(X, 3) : -1 : 1
    
    if strcmp(method, 'avg')
        
        error('avg is not supported.')
        
%         F(:, index_1) = mean(im2col(X(:, :, index_1), pool));
        
    elseif strcmp(method, 'max')
        
        [F(:, index_1), I] = max(im2col(X(:, :, index_1), pool));
        
        temp_2 = im2col(temp_1(:, :, index_1), pool);
        
        if isa(I, 'gpuArray')
            
            I = gather(I);
            
        end
        
        for index_2 = size(temp_2, 2) : -1 : 1
            
            G(index_2, index_1) = temp_2(I(index_2), index_2);
            
        end
        
    else
        
        error('%s is not supported.', method)
        
    end
    
end

d = size(F);
F = reshape(F, sqrt(d(1)), sqrt(d(1)), d(2));
F = F(1 : stride(1) : end, 1 : stride(2) : end, :);

if strcmp(method, 'avg')
    
    error('avg is not supported.')
    
%     G = [];
    
elseif strcmp(method, 'max')
    
    G = reshape(G, sqrt(d(1)), sqrt(d(1)), d(2));
    G = G(1 : stride(1) : end, 1 : stride(2) : end, :);
    G = single(G);
    
else
    
    error('%s is not supported.', method)
    
end

end

