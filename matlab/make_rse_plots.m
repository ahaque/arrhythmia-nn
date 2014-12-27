set(0,'DefaultAxesFontSize', 20)
set(0,'DefaultTextFontSize', 20)

CLASS = 'multiclass';
NUM_TO_PLOT = 40;

test_mse = csvread(strcat(CLASS,'_classification\test_mse.csv'));
train_mse = csvread(strcat(CLASS,'_classification\train_mse.csv'));
validate_mse = csvread(strcat(CLASS,'_classification\validate_mse.csv'));

test_mse(~test_mse) = nan;
train_mse(~train_mse) = nan;
validate_mse(~validate_mse) = nan;

figure;
plot(1:NUM_TO_PLOT,train_mse(1:6,1:NUM_TO_PLOT)','-^');
ylabel('Mean Squared Error (MSE)');
xlabel('Iteration');
legend('50','100','150','200','250','300','Location','NorthEast');
