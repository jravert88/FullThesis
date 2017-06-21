
function f = SOQPSK_frequency_pulse(N)
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