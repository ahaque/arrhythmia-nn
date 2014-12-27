clc;
clear;
CLASSIFICATION_STYLE = 1;
NUM_FEATURES = 279;

set(0,'DefaultAxesFontSize', 14)
set(0,'DefaultTextFontSize', 14)

training_sizes = [71	143	214	286	357	429	452];


data = csvread('data_clean_imputed.csv');

for T = 7:7
    X = data(1:training_sizes(T),1:NUM_FEATURES);
    y_label = data(1:training_sizes(T),end);

    if CLASSIFICATION_STYLE == 2
        % Multiclass classification
        y = zeros(length(data(1:training_sizes(T),1)),16);
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
    class_dist = tabulate(y_label);
    %figure
    %bar(class_dist(:,1),class_dist(:,2));
    %xlabel('Number of Records');
    %ylabel('Class Label');
    %xlim([0 17]);

    inputs = X';
    targets = y';

    %hiddenLayerSize = [10 25 50 100 150 200 250 300 350 400 500 600 700 800 900 1000 1500 2000];
    hiddenLayerSize = [100];
    mse_per_layer = zeros(1,length(hiddenLayerSize));
    for hls = 1:length(hiddenLayerSize)
        %net = patternnet(hiddenLayerSize(hls));
        net = patternnet([5000 10000 20000 1000]);
        % Set up Division of Data for Training, Validation, Testing
        net.divideParam.trainRatio = 62/100;
        net.divideParam.valRatio = 8/100;
        net.divideParam.testRatio = 30/100;
        net.trainFcn = 'trainscg';
        net.performFcn = 'mse';
        net.performParam.regularization = 0.01;

        % Train the Network
        [net,tr] = train(net,inputs,targets,'useGPU','no');

        % Test the Network
        outputs = net(inputs);
        errors = gsubtract(outputs,targets);
        %performance = perform(net,targets,outputs)
        mse_per_layer(hls) = mse(net,targets,outputs);

        % View the Network
        %view(net)
        dlmwrite('test_mse.csv',tr.tperf,'delimiter',',','-append');
        dlmwrite('validate_mse.csv',tr.vperf,'delimiter',',','-append');
        dlmwrite('train_mse.csv',tr.perf,'delimiter',',','-append');
        %figure, plotperform(tr)
        %figure, plottrainstate(tr)
        %figure, plotconfusion(targets,outputs)
        %figure, ploterrhist(errors)
        [val, idx] = max(outputs);

        % Plot the distributionn of predicted classes
        %class_dist2 = tabulate(idx);
        % figure
        %bar(class_dist2(:,1),class_dist2(:,2));
        split_index = round(training_sizes(T)*.7);
        test_size = training_sizes(T) - split_index;
        train_accuracy = sum(idx(1:split_index) - y_label(1:split_index)' ~= 0)/split_index;
        test_accuracy = sum(idx(split_index+1:end) - y_label(split_index+1:end)' ~= 0)/test_size;
        fprintf('NN Size: %d\tMSE: %d\n', hiddenLayerSize(hls), mse_per_layer(hls));
    end
end
plot(hiddenLayerSize, mse_per_layer,'-b^')
ylabel('Mean Squared Error (MSE)');
xlabel('Number of Layer 1 Neurons');