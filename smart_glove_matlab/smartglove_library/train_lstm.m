function [net] = train_lstm(features, labels)
%% Long short-term memory network training sequence

x = features;
y = labels;
%y_vector = ind2vec(y);
%x_cell = mat2cell(x, 1);
y_categories = categorical(y);


% Sort the data by length
numObservations = numel(x);

for i=1:numObservations
    sequence = x{i,1};
    [m, n] = size(sequence);
    sequenceLengths(i) = m;
end

[sequenceLengths,idx] = sort(sequenceLengths);
x = x(idx);
y_categories = y_categories(idx);

for iC = 1:numel(x)
  x{iC} = rot90(x{iC});
end


figure
plot(x{1}')
xlabel("Time Step")
title("Training Observation 1")
numFeatures = size(x{1},1);
legend("Feature " + string(1:numFeatures),'Location','northeastoutside')

figure
bar(sequenceLengths)
ylim([0 6000])
xlabel("Sequence")
ylabel("Length")
title("Sorted Data")

maxEpochs = 100;
miniBatchSize = 2;

options = trainingOptions('adam', ...
    'ExecutionEnvironment','auto', ...
    'GradientThreshold',1, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest', ...
    'Shuffle','every-epoch', ...
    'Verbose',1, ...
    'Plots','training-progress');


numFeatures = 11;
numHiddenUnits1 = 125;
numHiddenUnits2 = 100;
numClasses = 2;
layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits1,'OutputMode','sequence')
    dropoutLayer(0.2)
    lstmLayer(numHiddenUnits2,'OutputMode','last')
    dropoutLayer(0.2)
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];


net = trainNetwork(x, y_categories, layers, options);

end
