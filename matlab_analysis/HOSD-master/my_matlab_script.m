function [output_1, output_2] = my_matlab_script(input_1, input_2)

    clear hos;
    n_components_out = 1;
    lowpass= 90;    % Maybe hertz?
    Fs=1000;
    N=input_2;
    
    hos(n_components_out) = hosobject(3);
    hos.initialize(N,Fs,lowpass);
    hos.get_block(input_1,25,25,false);

    clf
    close
    
  output_1 = hos.waveform;
  %output_2 = full(hos.ximp(input_1));
  output_2 = full(hos.apply_filter(input_1));

end