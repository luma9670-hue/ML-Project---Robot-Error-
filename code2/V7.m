%% =========================================================
%  UR3 CobotOps – Random Forest Failure Predictor (code_1)
%  Baseline model: default hyperparameters, no class weighting,
%  no threshold tuning, default 0.5 decision boundary.
%% =========================================================
 
clc; clear; close all;
rng(42);
 
%% ── 1. Load data ──────────────────────────────────────────
fprintf('Loading dataset...\n');
data = readtable('dataset_02052023.xlsx');
 
% Drop non-feature columns
data(:, {'Num', 'Timestamp'}) = [];
 
%% ── 2. Features and targets ───────────────────────────────
targetGrip = 'grip_lost';
targetStop = 'Robot_ProtectiveStop';
 
featureCols = setdiff(data.Properties.VariableNames, ...
                      {targetGrip, targetStop}, 'stable');
 
X = table2array(data(:, featureCols));
y_grip = double(data.(targetGrip));
y_stop = double(data.(targetStop));
 
fprintf('Dataset  : %d rows x %d features\n', size(X,1), size(X,2));
fprintf('grip_lost positives     : %d / %d (%.1f%%)\n', ...
        sum(y_grip), numel(y_grip), 100*mean(y_grip));
fprintf('ProtectiveStop positives: %d / %d (%.1f%%)\n', ...
        sum(y_stop), numel(y_stop), 100*mean(y_stop));
 
%% ── 3. Stratified 80/20 train/test split ─────────────────
cv = cvpartition(y_grip, 'HoldOut', 0.20, 'Stratify', true);
 
X_train = X(training(cv), :);
X_test  = X(test(cv),     :);
 
yg_train = y_grip(training(cv));  yg_test = y_grip(test(cv));
ys_train = y_stop(training(cv));  ys_test = y_stop(test(cv));
 
fprintf('\nTrain: %d  |  Test: %d\n', sum(training(cv)), sum(test(cv)));
 
%% ── 4. Train baseline Random Forest models ────────────────
%  All hyperparameters at defaults:
%    NumTrees              = 100
%    MinLeafSize           = 1   (trees grow until leaves are pure)
%    NumPredictorsToSample = sqrt(numFeatures)
%    No class weighting
%    Decision threshold    = 0.5
 
fprintf('\nTraining baseline models...\n');
nTrees = 100;
 
mdl_grip = TreeBagger(nTrees, X_train, yg_train, ...
    'Method', 'classification');
 
mdl_stop = TreeBagger(nTrees, X_train, ys_train, ...
    'Method', 'classification');
 
%% ── 5. Predict on test set ────────────────────────────────
[~, score_grip] = predict(mdl_grip, X_test);
[~, score_stop] = predict(mdl_stop, X_test);
 
% Column index for class "1" (failure)
col_grip = find(strcmp(mdl_grip.ClassNames, '1'));
col_stop = find(strcmp(mdl_stop.ClassNames, '1'));
 
% Default threshold = 0.5
pred_grip = double(score_grip(:, col_grip) >= 0.5);
pred_stop = double(score_stop(:, col_stop) >= 0.5);
 
%% ── 6. Metrics ────────────────────────────────────────────
function printMetrics(yTrue, yPred, label)
    tp  = sum(yPred==1 & yTrue==1);
    tn  = sum(yPred==0 & yTrue==0);
    fp  = sum(yPred==1 & yTrue==0);
    fn  = sum(yPred==0 & yTrue==1);
    acc  = (tp+tn)/numel(yTrue);
    prec = tp / max(tp+fp, 1);
    rec  = tp / max(tp+fn, 1);
    f1   = 2*prec*rec / max(prec+rec, 1e-9);
    spec = tn / max(tn+fp, 1);
    fprintf('\n=== %s ===\n', label);
    fprintf('  Accuracy    : %.4f\n', acc);
    fprintf('  Precision   : %.4f\n', prec);
    fprintf('  Recall      : %.4f\n', rec);
    fprintf('  Specificity : %.4f\n', spec);
    fprintf('  F1-Score    : %.4f\n', f1);
    fprintf('  TP=%d  TN=%d  FP=%d  FN=%d\n', tp, tn, fp, fn);
end
 
printMetrics(yg_test, pred_grip, 'grip_lost');
printMetrics(ys_test, pred_stop, 'Robot_ProtectiveStop');
 
%% ── 7. Confusion matrix plots ─────────────────────────────
figure('Name','Baseline Confusion Matrices','Position',[100 100 1000 420]);
 
subplot(1,2,1);
cm = confusionmat(yg_test, pred_grip);
confusionchart(cm, {'No Failure','Grip Lost'}, ...
    'Title',         'grip\_lost  (Baseline – code\_1)', ...
    'RowSummary',    'row-normalized', ...
    'ColumnSummary', 'column-normalized');
 
subplot(1,2,2);
cm = confusionmat(ys_test, pred_stop);
confusionchart(cm, {'No Failure','Protective Stop'}, ...
    'Title',         'Robot\_ProtectiveStop  (Baseline – code\_1)', ...
    'RowSummary',    'row-normalized', ...
    'ColumnSummary', 'column-normalized');
 
sgtitle('UR3 CobotOps – Baseline Confusion Matrices', 'FontSize', 14);
saveas(gcf, 'confusion_matrix_v1.png');
fprintf('\nSaved: confusion_matrix_v1.png\n');