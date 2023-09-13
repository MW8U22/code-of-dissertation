% Define the folders from which to take the wav files and where to put the
% spectrograms
infolder='D:\final pro\3444 Set 1 HW_NoHW\NoHW\';
outfolder='C:\Users\王民舟\Desktop\final project\鲸鱼检索\鲸鱼检索-3444\Data\med_3444\nohw\';

% Collect the details of the wav files in the folder
dirdata=dir([infolder,'\*.wav']);

% FFT size for the spectrogram
Nfft=1024;

% Loop over all of the .wav files in the folder defined by "infolder"
for k=1:numel(dirdata)
    disp(k)                 % Display count

    % Get name of next file
    wfile=dirdata(k).name;

    % We create the full file names for the wav and png files, including their paths.

    % append the directory name to the input file name
    infile=[infolder,wfile];

    % Replace the .wav extension with .png, which defines the output (image) file name
    ind=strfind(wfile,'.wav');
    sfile=[wfile,'.png'];
    % append the directory name to the output file name
    outfile=[outfolder,sfile];

    % Read the audio data from the wav file
    [x,fs]=audioread(infile);
    
    % Compute the spectrogram
%    S = melSpectrogram(x,fs,NumBands=64,FFTLength=Nfft,Window=hanning(Nfft),FrequencyRange=[1 3000]);
    [S,f,t]=spectrogram(x,hanning(Nfft),round(0.75*Nfft),4*Nfft,fs);

    % Convert spectrogram amplitudes to a dB scale
    S=20*log10(abs(S));

    % Find the median value for each frequency bin (row in the spectrogram matrix S)
    Sm=median(S,2);

    % Subtract the median value from each point in the corresponding row of the spectrogram
    % The multiplication Sm*ones(1,size(S,2) creates a matrix the same size as S, with the same
    % value in a row, that value being the median.
    S=S-(Sm*ones(1,size(S,2)));

    % Plot the spectrogram against a mel frequency scale
    pcolor(t,log10(1+f/700),S),shading flat


    % Preparing the plot before saving it as a .png, removing axes and
    % fixing the colour scale

    % Turn off the labels
    set(gca,'XTick',[], 'YTick', [])

    % Fix the spectrogram colour amplitude scale
    caxis([-5 20])
   
    % Turn off the box around the plot
    set(gca,'box','off');

    % Make the picture big
    set(gca,'position',[0 0 1 1],'units','normalized')

    % Create a png file and call it the name defined by outfile
    print(outfile,'-dpng');
end