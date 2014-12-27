clc;
clear;
CLASSIFICATION_STYLE = 1;
NUM_FEATURES = 279;
NUM_COMPONENTS = 140;

set(0,'DefaultAxesFontSize', 12)
set(0,'DefaultTextFontSize', 12)


data = csvread('data_clean_imputed.csv');
m = length(data(:,1));
[U,S,V] = svd(data);

%X = data(:,1:NUM_FEATURES);
X = U(:,1:NUM_COMPONENTS);
y_label = data(:,end);
if CLASSIFICATION_STYLE == 2
    % Multiclass classification
    y = zeros(m,16);
    for i = 1:16
       y(y_label == i, i) = 1;
    end
end
if CLASSIFICATION_STYLE == 1
    % Binary classification
    y(y_label == 1,2) = 1;
    y(y_label > 1,1) = 1; % First column is the "true" and has arrithmia
end

% Plots distribution of input classes
% class_dist = tabulate(y_label);
% figure
% bar(class_dist(:,1),class_dist(:,2));
% ylabel('Number of Instances');
% xlabel('Class Label');
% xlim([0 17]);

inputs = X';
targets = y';

%hiddenLayerSize = [10 25 50 100 150 200 250 300 350 400 500 600 700 800 900 1000 1500 2000];
hiddenLayerSize = [100 100];
net = patternnet(hiddenLayerSize);
% Set up Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 0/100;
net.divideParam.testRatio = 30/100;
net.trainFcn = 'trainscg';
net.performFcn = 'mse';
net.performParam.regularization = .01;

% Train the Network
[net,tr] = train(net,inputs,targets,'useGPU','no');

% xg = nndata2gpu(inputs);
% tg = nndata2gpu(targets);
% net2 = configure(net,inputs,targets);
% net2 = train(net2,xg,tg);

% Test the Network
outputs = net(inputs);
errors = gsubtract(outputs,targets);
%performance = perform(net,targets,outputs)
mse = mse(net,targets,outputs);

[val, idx] = max(outputs);

% Plot the distributionn of predicted classes
%class_dist2 = tabulate(idx);
% figure
%bar(class_dist2(:,1),class_dist2(:,2));
% split_index = round(m*.7);
% test_size = m - split_index;
% train_accuracy = sum(idx(1:split_index) - y_label(1:split_index)' ~= 0)/split_index;
% test_accuracy = sum(idx(split_index+1:end) - y_label(split_index+1:end)' ~= 0)/test_size;