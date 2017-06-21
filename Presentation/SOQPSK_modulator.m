function y = SOQPSK_modulator(Nb,b)
% function y = SOQPSK_modulator(Nb,b)
% Nb = number of samples/bit
% b = bits (0,1)

f = SOQPSK_frequency_pulse_internal(Nb);
a = 2*b-1;
aa = zeros(size(a));
aa(1) = 0.5*(-1)*(a(1)+1);
aa(2) = 0.5*a(1)*(a(2)+1);
%
% The assumption for a(0) and a(-1) means
% initial phase state is 3*pi/2
% An additional -pi/4 phase shift is added to 
% generate nice looking I/Q eye diagrams
%
aa(3:end) = 0.5*a(2:end-1).*(a(3:end)-a(1:end-2));
aa(1:2:end) = -aa(1:2:end);
temp = zeros(1,Nb*(length(aa)-1)+1);
temp(1:Nb:end) = aa;
dphi = pi*conv(temp,f);
phi = filter([0 1],[1 -1],dphi) + 3*pi/2 - pi/4;
y = exp(1i*phi);
return;

function f = SOQPSK_frequency_pulse_internal(N)
% function f = SOQPSK_frequency_pulse_internal(N)
% N = samples/bit

rho = 0.7;
B = 1.25;
T1 = 1.5;
T2 = 0.5;

n1 = 0:2*N*T1-1;
n2 = 2*N*T1:2*N*(T1+T2);
nall = [n1 n2];

part1 = cos(pi*rho*B*nall/(2*N)) ./ (1-4*(rho*B*nall/(2*N)).^2);
part2 = sinc(B*nall/(2*N));
w = [ones(size(n1)) 0.5*(1+cos(pi/T2*(n2/(2*N)-T1)))];
temp = part1 .* part2 .* w;
f = [temp(end:-1:2) temp];
f = f*0.5/sum(f);
return;