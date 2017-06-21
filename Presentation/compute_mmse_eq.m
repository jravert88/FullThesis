function c = compute_mmse_eq(h,N1,N2,L1,L2,svar,nvar)
% function c = compute_mmse_eq(h,N1,N2,L1,L2,svar,nvar)
% h2 = channel coefficents h(n) for -N1<=n<=N2
% N1,N2 = channel span 
% L1,L2 = equalizer filter coefficient span 
% svar = variance of signal
% nvar = noise variance
% c = equalizer filter coefficients c(n) for -L1 <= n <= L2
% This file assumes the signal samples are uncorrelated.

g = zeros(1,L1+L2+1);
for idx = -L1:L2
    temp = -idx;
    if (temp >= -N1) && (temp <= N2)
        g(idx+L1+1) = h(temp+N1+1);
    end
end

G = zeros(L1+L2+1,N1+N2+L1+L2+1);
for idx = 1:L1+L2+1
    G(idx,idx:idx+N1+N2) = h(end:-1:1);
end

c = (G*G' + nvar/svar*eye(L1+L2+1))\g';