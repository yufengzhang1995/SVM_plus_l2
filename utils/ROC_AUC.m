function [X, Y, T, AUC, thresh, F1, sens, spec, acc] = ...
    ROC_AUC(scores, classes, method, n, DISPLAY_ROC, DISPLAY_ALL)
% Alexander Wood for BCIL 2019
%
%[X, Y, T, AUC] = ROC_AUC(scores, classes, n, DISPLAY_ROC, DISPLAY_ALL)
%
% Fast method of computing AUC, ROC. Splits into n different thresholds and
% computes for those thresholds. Works for binary classification. 
%
% Input scores and classes can be scalars or vectors.
%
% INPUT:
% scores........[double] The scores you assigned each point.
% classes.......[logical] The classes you assigned each point. (Ground
%               truth).
% n.............[integer, OPTIONAL, DEFAULT=10000] Number of thresholds to 
%               use. Should be large.
% method........[numeric, >=1, <=5, OPTIONAL, DEFAULT=1] Different methods 
%               for choosing a point on the ROC curve for which we compute 
%               the F1 scores, sensitivity,specificity and accuracy.
%               The methods choose the point on the ROC curve where...
%                  * 1: ... the F1-score is maximal.
%                  * 2: ... sensitivity+specificity is maximal.
%                  * 3: ... the ROC curve closest to (0,1).
%                  * 4: ... the sens. and specificity are about the same.
%                  * 5: ... the accuracy is maximal.
% DISPLAY_ROC...[logical, OPTIONAL, DEFAULT=true] Display the ROC plot
% DISPLAY_ALL...[logical, OPTIONAL, DEFAULT=false] Display other plots
%               (sens, spec, TPR, FPR, NPV, PPV, DICE/F1, etc)
% OUTPUT:
% X..............X-coordinates of ROC curve points
% Y..............Y-coordinates of ROC curve points
% T..............Threshold values used
% AUC............AUC for ROC curve
% thresh.........Optimal threshold based on 'method'
% F1.............Optimal F1 score
% sens...........Optimal sensitivity
% spec...........Optimal specificity
% acc............Optimal accuracy
%
% Note: scores and classes are both double vectors but they contain
% integers of 1 and 0
%
%%

% Default values
if ~exist('method','var')
    method = 1;
end
if ~exist('n','var')
    n = 10000;
end
if ~exist('DISPLAY_ROC','var')
    DISPLAY_ROC = 1;
end
if ~exist('DISPLAY_ALL', 'var')
    DISPLAY_ALL = 0;
end

% Input parsing
[scores, classes, method, n, DISPLAY_ROC, DISPLAY_ALL] = ...
    parse_inputs(scores, classes, method, n, DISPLAY_ROC, DISPLAY_ALL);

% Format scores and classes as scalars. 
scores  = scores(:);
classes = classes(:);

% Set up vectors to store the scores.
TP = zeros(n+1,1);
TN = zeros(n+1,1);
FP = zeros(n+1,1);
FN = zeros(n+1,1);
T  = [(1/(n-1))*(0:n-1)';1]; % thresholds

%% Compute the TP, FP, TN, FN for each threshold.

% Compute for each threshold, 0, 1/n, 2/n, ... , (n-1)/n
for i = 1:n
    positive = (scores > (i-1)/(n-1));
    TP(i) = sum( classes &  positive);
    TN(i) = sum(~classes & ~positive);
    FP(i) = sum(~classes &  positive);
    FN(i) = sum( classes & ~positive);
end

% Compute for ALL positive classfication.
TP(n+1) = sum(classes);
TN(n+1) = 0;
FP(n+1) = sum(~classes);
FN(n+1) = 0;

% Calculate the metrics for each threshold.
F1 = (2*TP)./(FN + 2*TP + FP); F1(isnan(F1))=0; % Only TN -> 0 F1-score
sens = TP./(TP + FN); %sens(isnan(sens)) = 0; % Only TN or FP -> 0 sens
spec = TN./(TN + FP);
acc  = (TP+TN)./(TP+TN+FP+FN); % As long as there is a datapoint this won't divide by zero
PPV  = TP./(TP+FP);  PPV(isnan(PPV))=1;
NPV  = TN./(TN+FN);  NPV(isnan(NPV))=1;

%% Compute AUC, optional plot ROC

% Compute (X,Y) = (TPR,FPR)
X = 1-spec; Y = sens;

% Sort, ascending.
f = sortrows([X, Y], [1 2]);
X = f(:,1);
Y = f(:,2);
% [X,perm] = sort(X,'ascend');
% Y = Y(perm);

% Compute dX operator for the right and left reimann sum.
dX = X(2:end)-X(1:end-1);

% Compute the left and right Reimann sums.
reimann_r = Y(2:end).*dX;
reimann_l = Y(1:end-1).*dX;

% Strictly increasing peicewise linear function (discrete case). So we
% add the area of the triangles between line and reimann rectangles to get
% AUC.
AUC = sum(reimann_l + 0.5*(reimann_r-reimann_l));

% Plot it. (Optional).
if DISPLAY_ROC
    figure(1); hold off; clf; hold on
    xlim([-0.02,1.02]); ylim([-0.02,1.02]);
    plot(X,Y,'LineWidth',1.5);
    line([0 1], [0 1], 'Color','red','LineStyle','--')
    legend('ROC','Random Guess','location','best');
    xlabel('False positive rate')
    ylabel('True positive rate')
    title('ROC Curve')
    text(0.6, 0.3, ['AUC: ' num2str(AUC)]) %text(0.6, 0.3, ['AUC: ' string(AUC)])
    hold off
end

% Optional Displays
if DISPLAY_ALL
    % Display sensitivity, specificity, F1-score
    hold off; figure(2); clf; hold on;
    xlim([-0.02,1.02]); ylim([-0.02,1.02]);
    plot(T,F1,'LineWidth',1.5)
    plot(T,sens,'g--','Color',[.66 .33 0],'LineWidth',1.5)
    plot(T,spec,'m--','Color',[.45 0 .55],'LineWidth',1.5);
    xlabel('Threshold')
    ylabel('Percent')
    legend('F1/Dice', 'Sens/TPR', 'Spec/FPR','location','best');
    
    % Display PPV, NPV, accuracy
    figure(3); hold off; clf; hold on;
    xlim([-0.02,1.02]); ylim([-0.02,1.02]);
    plot(T,acc,'LineWidth',1.5)
    plot(T,PPV,'--','Color',[.66 .33 0],'LineWidth',1.5)
    plot(T,NPV,'--','Color',[.45 0 .55],'LineWidth',1.5)
    xlabel('Threshold')
    ylabel('Percent')
    legend('Accuracy', 'PPV', 'NPV','location','best');
    
    % Display TP, TN, FP, FN counts
    figure(4); hold off; clf; hold on;
    M = 3*max([TP(:);FN(:);FP(:);TN(:)]);
    xlim([-0.02,1.02]); ylim([0,M+.02]);
    plot(T,TP,T,TN,T,FP,T,FN,'LineWidth',1.5);
    set(gca,'Yscale','log')
    xlabel('Threshold')
    ylabel('Count, Logaritihmic Scale')
    legend('TP', 'TN', 'FP', 'FN','location','best');
    disp(AUC)
end

% Compute the optimum threshold value. 
switch method
    case 1
        [~, thresh] = max(F1);
    case 2
        [~, thresh] = max(sens + spec);
    case 3
        [~, thresh] = max( (1-sens).^2 + (1-spec).^2 );
    case 4
        [~, thresh] = max(find(sens>spec, 1, 'first'));
    case 5
        [~, thresh] = max(acc);
end

% Set outputs based on thresh.
F1 = F1(thresh);
sens = sens(thresh);
spec = spec(thresh);
acc = acc(thresh);
end



%% INPUT VALIDATION
% Alexander Wood for BCIL 2019.
function [scores, classes, method, n, DISPLAY_ROC, DISPLAY_ALL] = ...
    parse_inputs(scores, classes, method, n, DISPLAY_ROC, DISPLAY_ALL)

validateattributes(classes,{'numeric','logical'},{'nonnan','nonempty',...
    'vector', 'binary'}, mfilename, 'classes');
classes = logical(classes);
classes = classes(:); % Set up as column, regardless of input size.

% Input scores must be same size as input classes and range from 0 to 1.
N = size(classes,1);
scores = scores(:); % Set up as colun, regardless of input size. 
validateattributes(scores, {'single','double'}, {'nonnan' 'nonempty', ...
    'vector', 'size', [N 1], 'nonnegative', '<=', 1}, mfilename, 'scores');

% Method should be integer 1, 2, 3, 4, or 5
validateattributes(method,{'numeric'},{'scalar', 'integer', ...
    '<=', 5, '>=', 1}, mfilename, 'method');

% n should be a positive integer.
validateattributes(n,{'numeric'},{'scalar','integer','positive'},...
    mfilename, 'n');

% DISPLAY_ROC should be 1 or 0
validateattributes(DISPLAY_ROC, {'numeric'},{'scalar','binary'},...
    mfilename,'DISPLAY_ROC');

% DISPLAY_ALL should be 1 or 0
validateattributes(DISPLAY_ALL, {'numeric'},{'scalar','binary'},...
    mfilename,'DISPLAY_ALL');

end