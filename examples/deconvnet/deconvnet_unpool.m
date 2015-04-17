function X_hat = deconvnet_unpool(d, F, G, pad)
%DECONVNET_UNPOOL Summary of this function goes here
%   Detailed explanation goes here

X_hat       = gpuArray(zeros(d + [sum(pad([1 2])) sum(pad([3 4])) 0], 'single'));
X_hat(G(:)) = F(:);
X_hat       = X_hat(pad(1) + 1 : end - pad(2), pad(3) + 1 : end - pad(4), :);

end

