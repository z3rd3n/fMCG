clear hos;
n_components_out = 1;
lowpass= 90;    % Maybe hertz?
Fs=1000;
N=545;

hos(n_components_out) = hosobject(3);
hos.initialize(N,Fs,lowpass);

%%% Train on the input data through a maximum of 25 iterations

mus = importdata('C:\LRZ\Cluster\Verena Wallner\bispectrale\to_matlab.txt','\t',0); 
for j = 2:1:3
    li = mus(:,j);
    clear hos;
    hos(n_components_out) = hosobject(3);
    hos.initialize(N,Fs,lowpass);
    hos.get_block(li,25);
    mus(:,j) = hos.apply_filter(mus(:,j));
    clf
    close
end

%mus = mus(:,2);
%plot(hos.waveform)


%hos.get_block(mus,25);
%recovered_ecg = hos.xrec(mus);
%filtered_ecg = hos.xthresh(ecgz_noise);



%plot(x,mus,x,recovered_ecg),legend('raw','recovered')
no_mother = hos.apply_filter(mus);
ximp=hos.ximp(mus(:,j));
save ('C:\LRZ\Cluster\Verena Wallner\bispectrale\no_mother.txt', 'mus', '-ascii')
