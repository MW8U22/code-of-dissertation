
audio_path = 'D:\final pro\Labels PACAP Set 1 3376\NOHW\';
file_list = dir(fullfile(audio_path, '*.wav'));


save_path = 'C:\Users\王民舟\Desktop\final project\鲸鱼检索\鲸鱼检索-3376\data\pcen_3376\nohw\';


for idx = 1:length(file_list)
    file = file_list(idx).name;
    full_path = fullfile(audio_path, file);
    

    [y, sr] = audioread(full_path);
    

    [S, f, t] = spectrogram(y, hanning(1024), 768, 4096, sr);
    S_dB = 20*log10(abs(S));

    S_PCEN = abs(pcen(S_dB, 0.025, 0.98, 2, 0.5, 1e-6));
    

    figure;
    pcolor(t, log10(1 + f/700), S_PCEN);
    shading flat;
    colormap('jet');  
    caxis([-5 20]);
    axis off;
  
    save_filename = [file(1:end-4), '_pcen.png'];
    save_filepath = fullfile(save_path, save_filename);
    
  
    saveas(gcf, save_filepath);
    close;
end


function S_PCEN = pcen(S, s, alpha, delta, r, eps)
    M = max(S, eps);
    smooth = (1 - s) * M + s * M;
    S_PCEN = (S ./ (eps + smooth.^alpha + delta)).^r;
end
