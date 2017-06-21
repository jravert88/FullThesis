clear all
load('h_all.mat')

N1 = 12;
N2 = 25;
Meq = 5;
nvar = 0;
for channel_select = [5] %1:11
    switch channel_select
        case 0
            h = 1;
            N1t = 0;
            N2t = 0;
            SNR = (0:15)';
        case 1
            if ~exist('h_all','var'), load h_all.mat, end
            h = h_all{channel_select};
            N1t = 1;
            N2t = 7;
            SNR = (0:16)';
        case 2
            if ~exist('h_all','var'), load h_all.mat, end
            h = h_all{channel_select};
            N1t = 2;
            N2t = 17;
            SNR = (0:21)';
        case 3
            if ~exist('h_all','var'), load h_all.mat, end
            h = h_all{channel_select};
            N1t = 1;
            N2t = 22;
            SNR = (0:21)';
        case 4
            if ~exist('h_all','var'), load h_all.mat, end
            h = h_all{channel_select};
            h = h(1:19);
            N1t = 6;
            N2t = 12;
            SNR = (0:21)';
        case 5
            if ~exist('h_all','var'), load h_all.mat, end
            h = h_all{channel_select};
            N1t = 1;
            N2t = 1;
            SNR = (0:28)';
        case 6
            if ~exist('h_all','var'), load h_all.mat, end
            h = h_all{channel_select};
            N1t = 1;
            N2t = 2;
            SNR = (0:18)';
        case 7
            if ~exist('h_all','var'), load h_all.mat, end
            h = h_all{channel_select};
            N1t = 0;
            N2t = 4;
            SNR = (0:21)';
        case 8
            if ~exist('h_all','var'), load h_all.mat, end
            h = h_all{channel_select};
            N1t = 2;
            N2t = 3;
            SNR = (0:21)';
        case 9
            if ~exist('h_all','var'), load h_all.mat, end
            h = h_all{channel_select};
            N1t = 1;
            N2t = 1;
            SNR = (0:16)';
        case 10
            if ~exist('h_all','var'), load h_all.mat, end
            h = h_all{channel_select};
            N1t = 2;
            N2t = 3;
            SNR = (0:27)';
        case 11
            if ~exist('h_all','var'), load h_all.mat, end
            h = h_all{channel_select};
            h = h(1:6);
            N1t = 3;
            N2t = 2;
            SNR = (0:34)';
        otherwise
            error('channel_select = %d not supported\n',channel_select);
    end
    h = h/sqrt(h'*h);
    %     h = [zeros(N1-N1t,1); h; zeros(N2-N2t,1)];3 5 6 7 9 11
    
    %     figure(channel_select); clf;
    %     subplot(211)
    %     stem(-N1t:N2t,abs(h)); grid on
    %     subplot(212)
    %     myNfft = 1024;
    %     FF = -0.5:1/myNfft:0.5-1/myNfft;
    %     HH = 10*log10(fftshift(abs(fft(h,myNfft)).^2));
    %     plot(FF,HH); grid on
    %     %     pause
    
    %     h = [zeros(N1-N1t,1); h; zeros(N2-N2t,1)];
    %     c = compute_mmse_eq(h,N1,N2,N1*Meq,N2*Meq,1,nvar);
    
    figure(1); clf;
    subplot(211)
    ff1 = stem(-N1t-1:N2t+1,[0; abs(h); 0]); grid on; hold on
    %     stem(-N1*Meq:N2*Meq,abs(c/sqrt(c'*c)));
    xlabel('samples (2 samples/bit)')
    ylabel('magnitude')
    subplot(212)
    myNfft = 1024;
    FF = -0.5:1/myNfft:0.5-1/myNfft;
    HH = 10*log10(fftshift(abs(fft(h,myNfft)).^2));
    ff2 = plot(FF,HH); grid on
    ylabel('magnitude (dB)')
    xlabel('frequency (cycles/sample)')
    
    
    figure(2); clf;
    subplot(211)
    ff3 = stem(-N1t-1:N2t+1,[0; abs(h); 0]); grid on; hold on
    %     stem(-N1*Meq:N2*Meq,abs(c/sqrt(c'*c)));
    xlabel('samples (2 samples/bit)')
    ylabel('magnitude')
    subplot(212)
    h2 = [zeros(N1-N1t,1); h; zeros(N2-N2t,1)];
    c = compute_mmse_eq(h2,N1,N2,N1*Meq,N2*Meq,1,nvar);
    myNfft = 1024;
    FF = -0.5:1/myNfft:0.5-1/myNfft;
    HH = 10*log10(fftshift(abs(fft(h,myNfft)).^2));
    CC = 10*log10(fftshift(abs(fft(c,myNfft)).^2));
    ff4 = plot(FF,HH,FF,CC); grid on
    ylabel('magnitude (dB)')
    xlabel('frequency (cycles/sample)')
    
    figure(3); clf;
    subplot(211)
    ff5 = stem(-N1t-1:N2t+1,[0; abs(h); 0]); grid on; hold on
    %     stem(-N1*Meq:N2*Meq,abs(c/sqrt(c'*c)));
    xlabel('samples (2 samples/bit)')
    ylabel('magnitude')
    subplot(212)
    h = [zeros(N1-N1t,1); h; zeros(N2-N2t,1)];
    c = compute_mmse_eq(h,N1,N2,N1*Meq,N2*Meq,1,nvar);
    myNfft = 1024;
    FF = -0.5:1/myNfft:0.5-1/myNfft;
    HH = 10*log10(fftshift(abs(fft(h,myNfft)).^2));
    CC = 10*log10(fftshift(abs(fft(c,myNfft)).^2));
    ff6 = plot(FF,HH,FF,CC,FF,CC+HH); grid on
    ylabel('magnitude (dB)')
    xlabel('frequency (cycles/sample)')
    
end

ff1(1).LineWidth =1;
ff1(1).Color = 'k';
for i = 1:1
ff2(i).LineWidth =1;
end
ff2(1).Color = 'b';

ff3(1).LineWidth =1;
ff3(1).Color = 'k';
for i = 1:2
ff4(i).LineWidth =1;
end
ff4(1).Color = 'b';
ff4(2).Color = 'r';

ff5(1).LineWidth =1;
ff5(1).Color = 'k';
for i = 1:3
ff6(i).LineWidth =1;
end
ff6(1).Color = 'b';
ff6(2).Color = 'r';
ff6(3).Color = 'm';

for myFigure = 1:3
    figure(myFigure)
    latexWidth = 5*0.8;
    latexHeight = 4*0.8;
    
    subplot(211)
    marge = axis;
    ax = gca;
    ax.FontName = 'Times New Roman';
    set(gca,'LineWidth',1)
    subplot(212)
    marge = axis;
    ax = gca;
    ax.FontName = 'Times New Roman';
    axis([marge(1:2) -20 20])
    set(gca,'LineWidth',1)
    
    ff = gcf;
    homer = ff.Units;
    ff.Units = 'inches';
    bart = ff.Position;
    ff.Position = [bart(1:2) latexWidth latexHeight];
    ff.PaperPositionMode = 'auto';
    ff.Units = homer;
    drawnow
    saveas(gcf,['multipath' num2str(myFigure)],'meta')
    
end



