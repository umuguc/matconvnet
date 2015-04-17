%%

close all;
clear all;
clc;

%%

MatConvNet_path = '~/GitHub/umuguc/matconvnet';
net_path        = '~/GitHub/umuguc/matconvnet/pretrained_models/imagenet-vgg-s.mat';

%%

run([MatConvNet_path '/matlab/vl_setupnn']);

%%

net = load(net_path);
net = vl_simplenn_move(net, 'gpu');

%%

X = imread('deconvnet_demo.jpeg');
X = imresize(X, net.normalization.imageSize([1 2]));
X = single(X);
X = X - net.normalization.averageImage;
X = gpuArray(X);

%%

activation  = [0 0];
feature_map = 0;
layer       = 0;

%%

X_hat = deconvnet_main(net, X, 'activation', activation, 'feature_map', feature_map, 'layer', layer);

%%

X_hat = X_hat - min(X_hat(:));
X_hat = X_hat / max(X_hat(:));

%%

imshow(X_hat);

