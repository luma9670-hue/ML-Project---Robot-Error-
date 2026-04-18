%% =========================================================
%  UR3 CobotOps - Multi-Output Random Forest Failure Predictor
%  Targets  : grip_lost | Robot_ProtectiveStop
%  Features : 20 sensor features (current, temperature, speed per joint)
%  Strategy : Two class-weighted TreeBagger models with OOB threshold
%             optimisation to maximise F1-score on each target.
%% =========================================================

clc; clear; close all;
rng(42);   % reproducibility

%% ── 1. LOAD DATA ──────────────────────────────────────────
fprintf('Loading dataset...\n');
data = readtable('dataset_02052023.xlsx');

% Identify the exact variable names MATLAB assigned
% (readtable sanitises spaces and special characters)
disp('Detected variable names:');
disp(data.Properties.VariableNames');

%% ── 2. FEATURE / TARGET EXTRACTION ───────────────────────
% Drop identifier and timestamp columns
dropCols = {'Num', 'Timestamp'};
data(:, dropCols) = [];

% Target columns (adjust names if MATLAB renamed them)
% MATLAB converts 'cycle ' → 'cycle_' and booleans stay as logical
targetGrip = 'grip_lost';
targetStop = 'Robot_ProtectiveStop';

% Build feature matrix X  (all columns except the two targets)
featureCols = setdiff(data.Properties.VariableNames, ...
                      {targetGrip, targetStop}, 'stable');
X = table2array(data(:, featureCols));
featureNames = featureCols;          % keep for importance plots

% Target vectors (force to double 0/1)
y_grip = double(data.(targetGrip));
y_stop = double(data.(targetStop));

fprintf('Dataset  : %d rows x %d features\n', size(X,1), size(X,2));
fprintf('grip_lost positives    : %d / %d  (%.1f%%)\n', ...
        sum(y_grip), numel(y_grip), 100*mean(y_grip));
fprintf('ProtectiveStop positives: %d / %d  (%.1f%%)\n', ...
        sum(y_stop), numel(y_stop), 100*mean(y_stop));

%% ── 3. STRATIFIED 80/20 TRAIN / TEST SPLIT ───────────────
% Use grip_lost for stratification; Stop follows naturally
cv = cvpartition(y_grip, 'HoldOut', 0.20, 'Stratify', true);

X_train = X(training(cv), :);
X_test  = X(test(cv),     :);

y_grip_train = y_grip(training(cv));
y_grip_test  = y_grip(test(cv));
y_stop_train = y_stop(training(cv));
y_stop_test  = y_stop(test(cv));

fprintf('\nTrain size: %d  |  Test size: %d\n', sum(training(cv)), sum(test(cv)));

%% ── 4. HELPER: CLASS WEIGHTS FOR IMBALANCED DATA ─────────
%  Weight = (total samples) / (num_classes * class_count)
%  This up-weights the rare failure class automatically.
function w = computeSampleWeights(y)
    n   = numel(y);
    n1  = sum(y == 1);
    n0  = n - n1;
    w   = zeros(n, 1);
    w(y == 1) = n / (2 * n1);
    w(y == 0) = n / (2 * n0);
end

%% ── 5. HYPERPARAMETER SEARCH (NumTrees & MinLeafSize) ─────
%  We sweep a small grid using OOB classification error as the
%  proxy metric, then pick the best combo for each target.
fprintf('\n--- Hyperparameter search ---\n');

numTreesGrid   = [100, 200, 300];
minLeafGrid    = [1, 5, 10, 20];

bestOOB_grip = inf;  bestNT_grip = 200; bestML_grip = 5;
bestOOB_stop = inf;  bestNT_stop = 200; bestML_stop = 5;

for nt = numTreesGrid
    for ml = minLeafGrid
        % --- grip_lost ---
        w = computeSampleWeights(y_grip_train);
        mdl = TreeBagger(nt, X_train, y_grip_train, ...
            'Method',       'classification', ...
            'MinLeafSize',  ml, ...
            'Weights',      w, ...
            'OOBPrediction','on', ...
            'NumPredictorsToSample', round(sqrt(size(X_train,2))));
        oob = oobError(mdl, 'Mode', 'ensemble');
        if oob < bestOOB_grip
            bestOOB_grip = oob;
            bestNT_grip  = nt;
            bestML_grip  = ml;
        end

        % --- Robot_ProtectiveStop ---
        w = computeSampleWeights(y_stop_train);
        mdl = TreeBagger(nt, X_train, y_stop_train, ...
            'Method',       'classification', ...
            'MinLeafSize',  ml, ...
            'Weights',      w, ...
            'OOBPrediction','on', ...
            'NumPredictorsToSample', round(sqrt(size(X_train,2))));
        oob = oobError(mdl, 'Mode', 'ensemble');
        if oob < bestOOB_stop
            bestOOB_stop = oob;
            bestNT_stop  = nt;
            bestML_stop  = ml;
        end
    end
end

fprintf('grip_lost   → NumTrees=%d, MinLeafSize=%d  (OOB=%.4f)\n', ...
        bestNT_grip, bestML_grip, bestOOB_grip);
fprintf('ProtStop    → NumTrees=%d, MinLeafSize=%d  (OOB=%.4f)\n', ...
        bestNT_stop, bestML_stop, bestOOB_stop);

%% ── 6. TRAIN FINAL MODELS ─────────────────────────────────
fprintf('\n--- Training final models ---\n');

w_grip = computeSampleWeights(y_grip_train);
mdl_grip = TreeBagger(bestNT_grip, X_train, y_grip_train, ...
    'Method',              'classification', ...
    'MinLeafSize',         bestML_grip, ...
    'Weights',             w_grip, ...
    'OOBPrediction',       'on', ...
    'OOBPredictorImportance', 'on', ...
    'NumPredictorsToSample', round(sqrt(size(X_train,2))));

w_stop = computeSampleWeights(y_stop_train);
mdl_stop = TreeBagger(bestNT_stop, X_train, y_stop_train, ...
    'Method',              'classification', ...
    'MinLeafSize',         bestML_stop, ...
    'Weights',             w_stop, ...
    'OOBPrediction',       'on', ...
    'OOBPredictorImportance', 'on', ...
    'NumPredictorsToSample', round(sqrt(size(X_train,2))));

%% ── 7. OOB THRESHOLD OPTIMISATION (maximise F1) ──────────
%  Use OOB probability scores on the training fold to tune the
%  classification threshold — avoids data leakage.

function thresh = optimiseThresholdF1(oobScores, yTrue)
    % oobScores : Nx1 probability of class "1"
    % yTrue     : Nx1 binary labels
    thresholds = 0.05 : 0.01 : 0.95;
    bestF1 = 0;  thresh = 0.5;
    for t = thresholds
        pred = double(oobScores >= t);
        tp = sum(pred == 1 & yTrue == 1);
        fp = sum(pred == 1 & yTrue == 0);
        fn = sum(pred == 0 & yTrue == 1);
        prec = tp / max(tp + fp, 1);
        rec  = tp / max(tp + fn, 1);
        f1   = 2 * prec * rec / max(prec + rec, 1e-9);
        if f1 > bestF1
            bestF1 = f1;
            thresh = t;
        end
    end
    fprintf('  Best threshold = %.2f  (F1 = %.4f)\n', thresh, bestF1);
end

fprintf('\n--- Optimising decision thresholds on OOB data ---\n');

% Extract OOB positive-class probabilities
oobProb_grip = oobPredict(mdl_grip);   % returns class labels; need scores
% Re-predict with probability output
[~, oobScore_grip] = oobPredict(mdl_grip);
[~, oobScore_stop] = oobPredict(mdl_stop);

% TreeBagger stores classes as {'0','1'}; column 2 = P(class="1")
classOrder_grip = mdl_grip.ClassNames;
classOrder_stop = mdl_stop.ClassNames;
col1_grip = find(strcmp(classOrder_grip, '1'));
col1_stop = find(strcmp(classOrder_stop, '1'));

fprintf('grip_lost threshold optimisation:\n');
thresh_grip = optimiseThresholdF1(oobScore_grip(:, col1_grip), y_grip_train);

fprintf('ProtectiveStop threshold optimisation:\n');
thresh_stop = optimiseThresholdF1(oobScore_stop(:, col1_stop), y_stop_train);

%% ── 8. PREDICT ON TEST SET ────────────────────────────────
[~, testScore_grip] = predict(mdl_grip, X_test);
[~, testScore_stop] = predict(mdl_stop, X_test);

pred_grip = double(testScore_grip(:, col1_grip) >= thresh_grip);
pred_stop = double(testScore_stop(:, col1_stop) >= thresh_stop);

%% ── 9. EVALUATION METRICS ─────────────────────────────────
function printMetrics(yTrue, yPred, label)
    tp = sum(yPred == 1 & yTrue == 1);
    tn = sum(yPred == 0 & yTrue == 0);
    fp = sum(yPred == 1 & yTrue == 0);
    fn = sum(yPred == 0 & yTrue == 1);
    acc  = (tp + tn) / numel(yTrue);
    prec = tp / max(tp + fp, 1);
    rec  = tp / max(tp + fn, 1);
    f1   = 2 * prec * rec / max(prec + rec, 1e-9);
    spec = tn / max(tn + fp, 1);
    fprintf('\n=== %s ===\n', label);
    fprintf('  Accuracy         : %.4f\n', acc);
    fprintf('  Precision        : %.4f\n', prec);
    fprintf('  Recall (Sens.)   : %.4f\n', rec);
    fprintf('  Specificity      : %.4f\n', spec);
    fprintf('  F1-Score         : %.4f\n', f1);
    fprintf('  TP=%d  TN=%d  FP=%d  FN=%d\n', tp, tn, fp, fn);
end

printMetrics(y_grip_test, pred_grip, 'grip_lost');
printMetrics(y_stop_test, pred_stop, 'Robot_ProtectiveStop');

%% ── 10. CONFUSION MATRIX PLOTS ────────────────────────────
figure('Name', 'Confusion Matrices', 'Position', [100 100 1000 420]);

subplot(1, 2, 1);
cm_grip = confusionmat(y_grip_test, pred_grip);
confusionchart(cm_grip, {'No Failure', 'Grip Lost'}, ...
    'Title',          'grip\_lost  –  Test Set', ...
    'RowSummary',     'row-normalized', ...
    'ColumnSummary',  'column-normalized');

subplot(1, 2, 2);
cm_stop = confusionmat(y_stop_test, pred_stop);
confusionchart(cm_stop, {'No Failure', 'Protective Stop'}, ...
    'Title',          'Robot\_ProtectiveStop  –  Test Set', ...
    'RowSummary',     'row-normalized', ...
    'ColumnSummary',  'column-normalized');

sgtitle('UR3 CobotOps – Random Forest Confusion Matrices', 'FontSize', 14);
saveas(gcf, 'confusion_matrices.png');

%% ── 11. FEATURE IMPORTANCE ────────────────────────────────
figure('Name', 'Feature Importance', 'Position', [100 100 1100 500]);

subplot(1, 2, 1);
imp_grip = mdl_grip.OOBPermutedPredictorDeltaError;
[sortedImp, idx] = sort(imp_grip, 'descend');
barh(sortedImp(end:-1:1));
yticks(1:numel(featureNames));
yticklabels(featureNames(idx(end:-1:1)));
xlabel('OOB Permutation Importance');
title('Feature Importance – grip\_lost');
grid on;

subplot(1, 2, 2);
imp_stop = mdl_stop.OOBPermutedPredictorDeltaError;
[sortedImp, idx] = sort(imp_stop, 'descend');
barh(sortedImp(end:-1:1));
yticks(1:numel(featureNames));
yticklabels(featureNames(idx(end:-1:1)));
xlabel('OOB Permutation Importance');
title('Feature Importance – Robot\_ProtectiveStop');
grid on;

sgtitle('UR3 CobotOps – OOB Permutation Feature Importance', 'FontSize', 14);
saveas(gcf, 'feature_importance.png');

%% ── 12. OOB ERROR CURVE (convergence) ─────────────────────
figure('Name', 'OOB Error Curve', 'Position', [100 100 700 400]);
plot(oobError(mdl_grip), 'b-', 'LineWidth', 1.5); hold on;
plot(oobError(mdl_stop), 'r-', 'LineWidth', 1.5);
xlabel('Number of Trees');
ylabel('OOB Classification Error');
legend('grip\_lost', 'Robot\_ProtectiveStop', 'Location', 'northeast');
title('OOB Error Convergence');
grid on;
saveas(gcf, 'oob_error_curve.png');

%% ── 13. SAVE MODELS ───────────────────────────────────────
save('ur3_rf_models.mat', 'mdl_grip', 'mdl_stop', ...
     'thresh_grip', 'thresh_stop', 'featureNames', 'featureCols');
fprintf('\nModels saved to ur3_rf_models.mat\n');

%% ── 14. INFERENCE EXAMPLE ─────────────────────────────────
% To predict on new data:
%
%   load('ur3_rf_models.mat');
%   newX = [your 20-feature row vector];
%   [~, s_grip] = predict(mdl_grip, newX);
%   [~, s_stop] = predict(mdl_stop, newX);
%   grip_prediction = s_grip(col1_grip) >= thresh_grip;   % 1 = failure
%   stop_prediction = s_stop(col1_stop) >= thresh_stop;   % 1 = failure
%
fprintf('\nDone.\n');