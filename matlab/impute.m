data = csvread('data\data_clean.csv');

X = data(:,1:end-1);
y = data(:,end);

impute = zeros(1,length(data(1,:)-1));
% For each column, compute mean from non-9999 values
for i = 1:length(data(1,:)-1)
   non9999 = find(data(:,i) ~= -9999);
   impute(i) = mean(data(non9999,i));
end

% Convert ? (or -9999s) to mean column values
for i = 1:length(data(:,1))
   for j = 1:length(data(1,:))
      if data(i,j) == -9999
         data(i,j) = impute(j);
      end
   end
end

csvwrite('data_clean_imputed.csv',data);