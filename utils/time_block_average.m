function averagedData = time_block_average()
% This function returns the average multiple pertubations in a single run. It
% only works for run # 105 at the moment!

dat = load('00105.mat');

% Divide the measurements into D blocks  of P samples.
P = 2^10; shift = 10; D = 7;

% Aprropriate time and frequency vector
t = linspace(0, (P - 1) / double(dat.NISampleRate), P)';

N = length(dat.SteerAngle);

% Simple but ugly algorithm for selecting blocks based on treshold on
% force input.
datFieldNames = fieldnames(dat);
numSamps = length(dat.SteerAngle);
for i = 1:length(datFieldNames)
    if length(dat.(datFieldNames{i})) == numSamps
        avg.(datFieldNames{i}) = zeros(P, D);
    end
end

avgFieldNames = fieldnames(avg);

j = 1; i = 1;
while i < N; % For every sample
    if abs(dat.PullForce(i)) > 20; % Check wether input treshhold is exceeded
        sel = (1:P) + i - shift;
        for k = 1:length(avgFieldNames)
            signal = dat.(avgFieldNames{k});
            avg.(avgFieldNames{k})(:, j) = signal(sel);
        end
        %disp([i,j])
        i = i + P; % Skip first P samples so that the blocks don't overlap.
        j = j + 1; % Next block
    end
    i = i + 1; % Next sample
    if j > D; break; end; % Enough is enough!!!!!
end

% Datacheck
figure(1);
title('Filtered Measurments');
for i = 1:length(avgFieldNames)
    subplot(length(avgFieldNames) / 3, 3, i);
    plot(t, avg.(avgFieldNames{i}), ':'); hold on;
    plot(t, mean(avg.(avgFieldNames{i}), 2), 'k-');
    ylabel(avgFieldNames{i});
end
xlabel('Time (s)');

for i = 1:length(avgFieldNames)
    averagedData.(avgFieldNames{i}) = mean(avg.(avgFieldNames{i}), 2);
end
