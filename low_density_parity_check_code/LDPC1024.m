clear all;
clc;
%% Init
M = 1024;            % Modulation Scheme: 1024QAM
codeLen = 64800;    % Codewords length
rate = 1 / 2;       % Coding rate
frmLen = codeLen * rate; % Frame length
snrset = 0 : 30;    % SNR
T = 10000;          % Iteration times
Es = 1;
m = log(M) / log(2);
hEnc = comm.LDPCEncoder(dvbs2ldpc(rate));
hMod = comm.RectangularQAMModulator('ModulationOrder', M, 'BitInput',true, 'NormalizationMethod', 'Average power');
hDec = comm.LDPCDecoder(dvbs2ldpc(rate));
bler = zeros(numel(snrset),1);      % Recode block error rate
goodput = zeros(numel(snrset),2);   % Calculate goodput based on bler

%% RUN
for ii = 1:numel(snrset)
    EsN0dB = snrset(ii);
    sigma2 = Es * 10^(-EsN0dB/10);
    sigma = sqrt(sigma2);
    hDemod = comm.RectangularQAMDemodulator('ModulationOrder', M, 'BitOutput',true, 'NormalizationMethod',...
        'Average power', 'DecisionMethod', 'Log-likelihood ratio', 'Variance', sigma2);
    bler(ii) = 0;
    for frmIdx = 1:T
        data = logical(randi([0 1], frmLen, 1));
        encodedData = step(hEnc, data);
        modSignal = step(hMod, encodedData);
        noise = sigma / sqrt(2) * (randn(size(modSignal)) + 1i * randn(size(modSignal)));
        receivedSignal = modSignal + noise;
        demodSignal = step(hDemod, receivedSignal);
        receivedBits = step(hDec, demodSignal);
        ber_1 = sum(data ~= receivedBits) / numel(data);
        if ber_1 ~= 0
            bler(ii) = bler(ii) + 1;
        end
    end
    bler(ii) = bler(ii) / T;
    goodput(ii,1) = m * rate * (1 - bler(ii));
end
plot(snrset,goodput(:,1));
grid on;
xlabel('SNR');
ylabel('goodput');