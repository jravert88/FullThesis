function [ out_data, out_x ] = minWindows(x,data,myWindowLength)
numWindows = floor(length(data)/myWindowLength);

out_x = zeros(1,numWindows);
out_data = zeros(1,numWindows);
for numWin = 1:numWindows
    myWindow = (numWin-1)*myWindowLength+1:numWin*myWindowLength;
    [A I] = min(data(myWindow));
    out_data(numWin)= A;
    out_x(numWin)   = x(min(myWindow)+I-1);
end
end

