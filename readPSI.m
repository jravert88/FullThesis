fileID = fopen('FreqTestFileFromMatlabPSI_r.txt','r');
formatSpec = '%f';
A = fscanf(fileID,formatSpec);
% whos A
fclose(fileID);
myAxis = -2:4/length(A):2-1/length(A);



% RR = pwelch(ss,blackman(nfft),nfft/2,nfft,'twosided')*1000000;

figure(1); clf;
plot(myAxis,10*log10((fftshift(A)))-4,'--')
% whos myAxis 

load('RR1.mat')
% rr = ifft(RR);
% rr = resample(rr,100,1);
% RR = fft(RR);
hold on
myAxis = -4:8/length(RR):4-1/length(RR);
plot(myAxis,10*log10(fftshift(RR))-65-5)
grid on
hold off

% marge = axis;
% axis([-2 2 -50 10])

latexWidth = 5;
latexHeight = 4;
ff = gcf;
homer = ff.Units;
ff.Units = 'inches';
bart = ff.Position;
ff.Position = [bart(1:2) latexWidth latexHeight];
ff.PaperPositionMode = 'auto';
ff.Units = homer;
xlabel('cycles/bit')
ylabel('Magintude (dB)')
% title('SOQPSK-TG Constellation')
legend('\Psi','SOQPSK Power Spectrum','Location','NorthWest')
% print(ff, '-depsc', ['FDE2_spectrum_PSI']) %save as eps a 






data_bits = randi([0 1],2^14*2,1);
data_bits = data_bits(1:6144);
ss = SOQPSK_modulator(10,data_bits);
s = ss(100:5:end-100);
[myXcorr lags] = xcorr(s);
myXcorr = myXcorr/max(abs(myXcorr));

figure(2); clf
subplot(211)
plot(lags,real(myXcorr)); grid on; hold on
myI = find(real(myXcorr)>0.04);
plot(lags(myI),real(myXcorr(myI)),'bo'); grid on; hold on
% axis
% axis
subplot(212)
plot(lags,imag(myXcorr),'b'); grid on; hold on
plot(lags,imag(myXcorr),'bo'); grid on; hold on
% axis = marge;
