set(0,'DefaultAxesFontSize', 20)
set(0,'DefaultTextFontSize', 20)

%{
axis_binary =[5	10	20	25	30	40	50	60	70	80	90	100	110	120	130	140	150	160	170	180	190	200	210	220	230	240	250	260	270	280	290];
train_binary = [0.994 	0.997	0.994	0.981	1	0.981	0.987	0.997	0.997	1	1	1	1	1	1	1	1	1	1	1	1	1	0.997	1	1	1	1	1	1	1	1];
test_binary = [0.632	0.61	0.662	0.728	0.699	0.779	0.743	0.75	0.699	0.868	0.897	0.904	0.904	0.912	0.897	0.919	0.779	0.875	0.816	0.772	0.772	0.757	0.816	0.684	0.779	0.75	0.787	0.669	0.713	0.721	0.684];

axis_multi = [5	10	20	30	40	50	60	70	80	90	100	110	120	130	140	150	160	170	180	190	200	210	220	230	240	250	260	270	280	290];
train_multi = [0.987	0.981	0.987	0.997	0.991	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1	1];
test_multi = [0.507	0.507	0.654	0.622	0.603	0.551	0.596	0.588	0.684	0.757	0.757	0.684	0.743	0.728	0.743	0.647	0.721	0.699	0.669	0.647	0.669	0.632	0.61	0.684	0.662	0.647	0.61	0.61	0.61	0.574];

figure
plot(axis_binary, test_binary, '-^b',...
    axis_multi, test_multi, '-^r');
xlabel('Principal Components');
ylabel('Accuracy');
legend('Binary (Test)','Multi (Test)','Location','NorthEast');

tix=get(gca,'ytick')';
set(gca,'yticklabel',num2str(tix,'%.2f'))
%ylim([0 1]);


reg = [0.0001	0.0005	0.001	0.005	0.01	0.05	0.1	0.5];
reg_test = [0.743	0.743	0.787	0.765	0.757	0.765	0.765	0.772];
reg_train = [1 1 1 1 1 1 1 1];

figure
semilogx(reg, reg_train, '-^b',...
    reg, reg_test, '-^r');
xlabel('Regularization Parameter');
ylabel('Accuracy');
legend('Multi (Train)','Multi (Test)','Location','NorthEast');
tix=get(gca,'ytick')';
set(gca,'yticklabel',num2str(tix,'%.2f'))
%}

figure
runtime = [6.3 3.5; 27.8 31.2]';
bar(runtime)
legend('GPU@1.2GHz (640 cores)','CPU@3.4GHz (6 cores)','Location','NorthEast');
ylim([0 50]);
set(gca,'XTickLabel',{'Binary', 'Multi-Class'})
xlabel('Classification Task');
ylabel('Training Time (seconds)');