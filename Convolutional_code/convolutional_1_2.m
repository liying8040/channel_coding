clear; 
clc;
snrdB = 0 : 35;
M = 128;                % Modulation order
k = log2(M);            % Bits per symbol
T =10000;               % Iteration times
rate = 1/2;             % Coding rate
FrmLen = 350;           % Number of bit per frame
berEstSoft = zeros(numel(snrdB));
rng default
trellis = poly2trellis(7, [171 133]);
tbl = 32;

hConvEnc = comm.ConvolutionalEncoder(trellis); 
hVitDecSoft = comm.ViterbiDecoder(trellis, 'InputFormat', 'Unquantized', ...
    'SoftInputWordLength', 8, 'TracebackDepth', tbl, ...
    'TerminationMethod', 'Continuous');

hMod = comm.RectangularQAMModulator('ModulationOrder', M, 'BitInput', true);
hDemodSoft = comm.RectangularQAMDemodulator('ModulationOrder', M, 'BitOutput', 1,...
    'DecisionMethod', 'Approximate Log-likelihood ratio');

bler = zeros(numel(snrdB),1);
goodput = zeros(numel(snrdB),1);
for ii = 1:numel(snrdB)
    for frmIdx = 1 : T
        dataIn = randi([0 1], FrmLen, 1);
        dataEnc = step(hConvEnc, dataIn);
        txSig = step(hMod, dataEnc);
        rxSig = awgn(txSig, snrdB(ii), 'measured');
        rxDataSoft = step(hDemodSoft, rxSig);
        dataSoft = step(hVitDecSoft, rxDataSoft);
        numErrsInFrameSoft = biterr(dataIn(1:end-tbl), dataSoft(tbl+1:end));
        if numErrsInFrameSoft ~= 0
            bler(ii) = bler(ii) + 1;
        end
    end
    bler(ii) = bler(ii) / T;
    goodput(ii,1) = k * rate *(1-bler(ii));
end
plot(snrdB, goodput(:,1), '-*r');
grid on;