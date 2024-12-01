
%% read data

T = readtable('testingData_Task2');

All_data_Test_T2 = T{:,:};
clear T;

%%

%load Test_T2_f39.mat;
%segment_size = 2500;

%load Test_T2_f39_1sec.mat;
%load Models_1sec.mat
%segment_size = 250;

load Test_T2_f39_100ss.mat;
load Models_100ss.mat
segment_size = 100;

time = All_data_Test_T2(:,1); sub_idx = All_data_Test_T2(:,5);

unique_sub_idx = unique(sub_idx,'stable');

Fs = 250;

%% predict

test_result = zeros(4,size(time,1));

N_subjects = size(unique_sub_idx,1);

cum_timesteps = 0;
cum_N_segments = 0;

% subject wise analysis
for ii = 1:N_subjects
    curr_sub = unique_sub_idx(ii);
    curr_time = time(sub_idx == curr_sub);

    curr_timesteps = size(curr_time,1);

    curr_N_segments = floor(size(curr_time,1)/segment_size);

    test_result(1,cum_timesteps+1:cum_timesteps+curr_timesteps) = cum_timesteps:cum_timesteps+curr_timesteps-1;
    test_result(2,cum_timesteps+1:cum_timesteps+curr_timesteps) = curr_sub;
    test_result(3,cum_timesteps+1:cum_timesteps+curr_timesteps) = curr_time;

    curr_feat_test_T2 = All_feat_test_T2(:,cum_N_segments+1:cum_N_segments+curr_N_segments);

    % use model to predict
    %[segment_label,scores] = trainedModel_NN_Bilayered.predictFcn(curr_feat_test_T2);
    %[segment_label,scores] = trainedModel_NN_Bilayered_1sec.predictFcn(curr_feat_test_T2);
    %[segment_label,scores] = trainedModel_NN_Bilayered_100ss.predictFcn(curr_feat_test_T2);
    [segment_label,scores] = trainedModel_NN_Wide_100ss.predictFcn(curr_feat_test_T2);

    % smoothen the labels (only good for one direction change)
    %filter_L = 3;
    %filtered_segment_label = filter(ones(1,filter_L)/3,1,segment_label);
    %filtered_segment_label(1:filter_L-1) = segment_label(1:filter_L-1);
    %filtered_segment_label = filtered_segment_label > 0.5;

    % use median filtering
    filter_L = 11;
    segment_label_padded = [ones(filter_L-1,1)*segment_label(1);segment_label;ones(filter_L-1,1)*segment_label(end)];
    filtered_segment_label = medfilt1(segment_label_padded,filter_L,'truncate');
    filtered_segment_label = filtered_segment_label(filter_L:filter_L+curr_N_segments-1);

    % repeat labels
    timestep_label = repelem(filtered_segment_label,segment_size);
    
    last_label = timestep_label(end);
    extend_timestep_label = [timestep_label;last_label*ones(curr_timesteps - size(timestep_label,1),1)];

    test_result(4,cum_timesteps+1:cum_timesteps+curr_timesteps) = extend_timestep_label;

    cum_timesteps = cum_timesteps + curr_timesteps;
    cum_N_segments = cum_N_segments + curr_N_segments;
end


%% write csv file

% writematrix(test_result','Santosh_T2_try3_1sec.csv','WriteMode','append');

%writematrix(test_result','Santosh_T2_try3A_1sec_MedFilter_BiLayered.csv','WriteMode','append');

%writematrix(test_result','Santosh_T2_try4_100ss_MedFilter_BiLayered.csv','WriteMode','append');

writematrix(test_result','Santosh_T2_try5_100ss_MedFilter_Wide.csv','WriteMode','append');


