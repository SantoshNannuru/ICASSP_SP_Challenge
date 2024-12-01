%%
clear;
T = readtable('trainingData_challenge1');

All_data_train_T1 = T{:,:};
clear T;

%%
x = All_data_train_T1(:,3); y = All_data_train_T1(:,4); z = All_data_train_T1(:,5);
time = All_data_train_T1(:,2); sub_idx = All_data_train_T1(:,6);
segment_idx = All_data_train_T1(:,7);  label = All_data_train_T1(:,8);

unique_sub_idx = unique(sub_idx);

Fs = 250;

%%

segment_size = 100;

% individual data segments
x_input = reshape(x,segment_size,size(x,1)/segment_size);
y_input = reshape(y,segment_size,size(y,1)/segment_size);
z_input = reshape(z,segment_size,size(z,1)/segment_size);

true_label = label(1:segment_size:end);

% variance, skewness, kurtosis
x_variance = var(x_input,0,1);
x_skewness = skewness(x_input,0,1);
x_kurtosis = kurtosis(x_input,0,1);

y_variance = var(y_input,0,1);
y_skewness = skewness(y_input,0,1);
y_kurtosis = kurtosis(y_input,0,1);

z_variance = var(z_input,0,1);
z_skewness = skewness(z_input,0,1);
z_kurtosis = kurtosis(z_input,0,1);

% DFT
L = 250;
X_DFT = fft(x_input,L,1);
Y_DFT = fft(y_input,L,1);
Z_DFT = fft(z_input,L,1);

% DFT (positive frequencies)
X_DFT = abs(X_DFT(2:L/2+1,:)/L);
Y_DFT = abs(Y_DFT(2:L/2+1,:)/L);
Z_DFT = abs(Z_DFT(2:L/2+1,:)/L);

% DFT features
Nband = 10;
X_feat = zeros(Nband,size(X_DFT,2));
Y_feat = zeros(Nband,size(X_DFT,2));
Z_feat = zeros(Nband,size(X_DFT,2));

for ii = 1:Nband
    X_feat(ii,:) = vecnorm(X_DFT(1+(ii-1)*L/2/Nband:ii*L/2/Nband,:),2,1);
    Y_feat(ii,:) = vecnorm(Y_DFT(1+(ii-1)*L/2/Nband:ii*L/2/Nband,:),2,1);
    Z_feat(ii,:) = vecnorm(Z_DFT(1+(ii-1)*L/2/Nband:ii*L/2/Nband,:),2,1);
end

%%

All_feat_train_T1 = [X_feat; Y_feat; Z_feat];

All_feat_train_T1 = [All_feat_train_T1; x_variance; x_skewness; x_kurtosis];

All_feat_train_T1 = [All_feat_train_T1; y_variance; y_skewness; y_kurtosis];

All_feat_train_T1 = [All_feat_train_T1; z_variance; z_skewness; z_kurtosis];

