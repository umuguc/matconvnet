function deconvnet = convnet_to_deconvnet(convnet)
%CONVNET_TO_DECONVNET Summary of this function goes here
%   Detailed explanation goes here

deconvnet = convnet;

for index = 1 : length(deconvnet.layers)
    
    switch convnet.layers{index}.type
        
        case 'conv'
            
            if isa(deconvnet.layers{index}.biases, 'gpuArray')
                
                deconvnet.layers{index}.biases = gpuArray(zeros(1, size(deconvnet.layers{index}.filters, 3), 'single'));
                
            else
                
                deconvnet.layers{index}.biases = zeros(1, size(deconvnet.layers{index}.filters, 3), 'single');
                
            end
            
            deconvnet.layers{index}.filters = flip(deconvnet.layers{index}.filters, 1);
            deconvnet.layers{index}.filters = flip(deconvnet.layers{index}.filters, 2);
            deconvnet.layers{index}.filters = permute(deconvnet.layers{index}.filters, [1 2 4 3]);
            deconvnet.layers{index}.type    = 'deconvnet_conv';
            
        case 'relu'
            
            deconvnet.layers{index}.type = 'deconvnet_relu';
            
        case 'normalize'
            
            deconvnet.layers{index}.type = 'deconvnet_normalize';
            
        case 'pool'
            
            deconvnet.layers{index}.type = 'deconvnet_unpool';
            
        otherwise
            
            error('Unknown layer type %s', l.type);
            
    end
    
end

deconvnet.layers = deconvnet.layers(end : -1 : 1);

end

