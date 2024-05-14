clc;clear

lowpass= 70;    

Fs=1000;
N = 2*Fs;
data = load('P044_R001_33channel.mat').data;
data = data(:,6);

%%% Initialize the HOS object
clear hos;
hos = hosobject(3);
hos.initialize(1400,Fs,lowpass);

%%% Train on the input data through a maximum of 25 iterations
hos.get_block(data(10*Fs:end),30);
recovered_ecg = hos.xrec(data);
filtered_ecg = hos.xthresh(data);

output_1 = hos.waveform;

output_2 = full(hos.apply_filter(data));
figure

plot(output_1);


       



