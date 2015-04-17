function X_hat = deconvnet_main(net, X, varargin)
%DECONVNET_MAIN Summary of this function goes here
%   Detailed explanation goes here

conf.activation  = [0 0];
conf.feature_map = 0;
conf.layer       = 0;

conf = vl_argparse(conf, varargin);

if nargin > 2
    
    assert(numel(conf.activation) == 2);
    assert(all(conf.activation >= 0));
    assert(conf.feature_map >= 0);
    assert(conf.layer >= 0 && conf.layer <= length(net.layers));
    
end

net.layers(conf.layer + 1 : end) = [];

L = numel(net.layers);
Y = struct('d', cell(1, L + 1), 'F', cell(1, L + 1), 'G', cell(1, L + 1));

Y(1).F = X;
Y(1).d = size(Y(1).F);

for index = 1 : L
    
    switch net.layers{index}.type
        
        case 'conv'
            
            Y(index + 1).F = vl_nnconv(Y(index).F, net.layers{index}.filters, net.layers{index}.biases, 'pad', net.layers{index}.pad, 'stride', net.layers{index}.stride);
            Y(index + 1).d = size(Y(index + 1).F);
            
        case 'relu'
            
            Y(index + 1).F = vl_nnrelu(Y(index).F);
            Y(index + 1).d = size(Y(index + 1).F);
            
        case 'normalize'
            
            Y(index + 1).F = vl_nnnormalize(Y(index).F, net.layers{index}.param);
            Y(index + 1).d = size(Y(index + 1).F);
            
        case 'pool'
            
            [Y(index + 1).F, Y(index + 1).G] = deconvnet_pool(Y(index).F, net.layers{index}.method, net.layers{index}.pad, net.layers{index}.pool, net.layers{index}.stride);
            Y(index + 1).d = size(Y(index + 1).F);
            
        otherwise
            
            error('%s is not supported', net.layers{index}.type);
            
    end
    
    Y(index).F = [];
    
    if isa(X, 'gpuArray')
        
        wait(gpuDevice);
        
    end
    
end

net = convnet_to_deconvnet(net);
d   = size(Y(end).F);

if conf.feature_map > 0
    
    assert(conf.feature_map <= d(3))
    
    Y(end).F(:, :, [1 : conf.feature_map - 1 conf.feature_map + 1 : end]) = 0;
    
end

if conf.activation(1) > 0
    
    assert(conf.activation(1) <= d(1))
    
    Y(end).F([1 : conf.activation(1) - 1 conf.activation(1) + 1 : end], :, :) = 0;
    
end

if conf.activation(1) > 0
    
    assert(conf.activation(2) <= d(2))
    
    Y(end).F(:, [1 : conf.activation(2) - 1 conf.activation(2) + 1 : end], :) = 0;
    
end

Y = Y(end : -1 : 1);

for index = 1 : L
    
    switch net.layers{index}.type
        
        case 'deconvnet_conv'
            
            temp_1 = Y(index).F;
            d      = size(net.layers{index}.filters);
            
            if net.layers{index}.stride(1) > 1
                
                temp_1 = upsample(temp_1, net.layers{index}.stride(1));
                temp_1 = temp_1(1 : Y(index + 1).d(1) + sum(net.layers{index}.pad([1 2])) - d(1) + 1, :, :);
                
            end
            
            if net.layers{index}.stride(2) > 1
                
                temp_1 = permute(temp_1, [2 1 3]);
                temp_1 = upsample(temp_1, net.layers{index}.stride(2));
                temp_1 = permute(temp_1, [2 1 3]);
                temp_1 = temp_1(:, 1 : Y(index + 1).d(2) + sum(net.layers{index}.pad([3 4])) - d(2) + 1, :);
                
            end
            
            temp_1 = padarray(temp_1, [d([1 2]) - 1 0]);
            
            Y(index + 1).F = vl_nnconv(temp_1, net.layers{index}.filters, net.layers{index}.biases);
            Y(index + 1).F = Y(index + 1).F(net.layers{index}.pad(1) + 1 : end - net.layers{index}.pad(2), net.layers{index}.pad(3) + 1 : end - net.layers{index}.pad(4), :);
            
        case 'deconvnet_relu'
            
            Y(index + 1).F = vl_nnrelu(Y(index).F);
            
        case 'deconvnet_normalize'
            
            Y(index + 1).F = Y(index).F;
            
        case 'deconvnet_unpool'
            
            Y(index + 1).F = deconvnet_unpool(Y(index + 1).d, Y(index).F, Y(index).G, net.layers{index}.pad);
            
        otherwise
            
            error('%s is not supported', net.layers{index}.type);
            
    end
    
    Y(index).F = [];
    
    if isa(X, 'gpuArray')
        
        wait(gpuDevice);
        
    end
    
end

X_hat = Y(end).F;

end

