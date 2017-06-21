close all
clear all

figure(1); clf
convLength = 1:2^14+1024;
L_m = [186 23 186+23-1];
for loops = 1:length(L_m)
L = L_m(loops);
N = convLength-L+1;
numFLOPSperConvOutput = 7*L; % (4 multiplies, 2 adds)*L + L(1 add for the summation)

convFlops = convLength.*numFLOPSperConvOutput;

fftFlops = 0*convFlops;
for i = 1:length(N)
    Nfft = 2^nextpow2(L+N(i)-1);
    fftFlops(i) = 5*Nfft*log2(Nfft) + 5*Nfft*log2(Nfft) + 6*Nfft + 5*Nfft*log2(Nfft);
end




    latexWidth = 5*0.8;
    latexHeight = 4*0.8;
figure(loops)
ff1 = plot(N,convFlops,'k-', N,fftFlops,'k--'); grid on
for i = 1:2
ff1(i).LineWidth =1;
end
marge = axis;
axis([0 16497 marge(3:4)])
ax = gca;
legend('Time Domain','Frequency Domain','Location', 'NorthWest')
% title(['Convolution with Filter Length ' num2str(L)])
xlabel('signal length')
ylabel('flops')
ax.FontName = 'Times New Roman';

ff = gcf;
homer = ff.Units;
ff.Units = 'inches';
bart = ff.Position;
ff.Position = [bart(1:2) latexWidth latexHeight];
ff.PaperPositionMode = 'auto';
ff.Units = homer;
drawnow
% print(ff, '-depsc', ['Theory' num2str(L) 'Tap_flops']) %save as eps a 
marge = axis;
    ax = gca;
    ax.FontName = 'Times New Roman';
    set(gca,'LineWidth',1)
    
    ff = gcf;
    homer = ff.Units;
    ff.Units = 'inches';
    bart = ff.Position;
    ff.Position = [bart(1:2) latexWidth latexHeight];
    ff.PaperPositionMode = 'auto';
    ff.Units = homer;
    drawnow
    saveas(gcf,['Theory' num2str(L) 'Tap_flops'],'meta')
end




    latexWidth = 5*0.8;
    latexHeight = 4*0.8;
figure(3)

L = 1:383;
N = 12672;
numFLOPSperConvOutput = 7*L; % (4 multiplies, 2 adds)*L + L(1 add for the summation)

convFlops = (N+L-1).*numFLOPSperConvOutput;

fftFlops = 0*convFlops;
for i = 1:length(L)
    Nfft = 2^nextpow2(L(i)+N-1);
    fftFlops(i) = 5*Nfft*log2(Nfft) + 5*Nfft*log2(Nfft) + 6*Nfft + 5*Nfft*log2(Nfft);
end
% return

ff1 = plot(L,convFlops,'k-', L,fftFlops,'k--'); grid on
for i = 1:2
ff1(i).LineWidth =1;
end
marge = axis;
axis([1 381 marge(3:4)])
ax = gca;
legend('Time Domain','Frequency Domain','Location', 'NorthWest')
% title(['Convolution with Filter Length ' num2str(L)])
xlabel('filter length')
ylabel('flops')
ax.FontName = 'Times New Roman';

ff = gcf;
homer = ff.Units;
ff.Units = 'inches';
bart = ff.Position;
ff.Position = [bart(1:2) latexWidth latexHeight];
ff.PaperPositionMode = 'auto';
ff.Units = homer;
drawnow

    marge = axis;
    ax = gca;
    ax.FontName = 'Times New Roman';
    set(gca,'LineWidth',1)
    
    ff = gcf;
    homer = ff.Units;
    ff.Units = 'inches';
    bart = ff.Position;
    ff.Position = [bart(1:2) latexWidth latexHeight];
    ff.PaperPositionMode = 'auto';
    ff.Units = homer;
    drawnow
    saveas(gcf,['Theory' num2str(N) 'signal_flops'],'meta')
% print(ff, '-depsc', ['Theory' num2str(N) 'signal_flops']) %save as eps a 

