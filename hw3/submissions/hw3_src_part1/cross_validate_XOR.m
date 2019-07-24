
clearvars;

load apples;
load oranges;
load apples2;
load oranges2;

% set some relevant parameters
eta=.1; % learning rate
alpha=.5;  % momentum term
num_trials=1000;
kfolds = 5;

sigmoid = @(x) 1 ./ (1 + exp(-x));

% initialize data array
data=[apples apples2 oranges oranges2];
[N, K]=size(data);
labels = [ones(1,K/2) zeros(1,K/2)];

% ---------------CALCULATE STRATIFIED CROSSVAL SPLITS---------------
% You need to calculate matrices 'train_inds' and 'test_inds'
% These matrices will be used to index into the full data to get train and
% test splits during cross-validation.
% Some code is provided for you to get started. You can feel free to use or
% not use it so long as you calculate 'train_inds' and 'test_inds'
% 
train_inds = zeros(kfolds, K-K/kfolds);
test_inds = zeros(kfolds, K/kfolds);
rand_inds_0s = randperm(20);
rand_inds_1s = 20 + randperm(20);
rand_inds_all = [rand_inds_0s, rand_inds_1s];
for fold = 1:kfolds
    inds = K/(kfolds*2)*(fold-1)+1:K/(kfolds*2)*(fold);
    test_inds(fold,:) = [rand_inds_0s(inds), rand_inds_1s(inds)];
    train_inds(fold,:) = setdiff(rand_inds_all, [inds, 20+inds]);
end
% ---------------end CALCULATE STRATIFIED CROSSVAL SPLITS---------------


colors = jet(2);

train_error = zeros(kfolds,1);
test_error = zeros(kfolds,1);
for fold = 1:kfolds
    train_data = data(:,train_inds(fold,:));
    train_labels = labels(:,train_inds(fold,:));
    test_data = data(:,test_inds(fold,:));
    test_labels = labels(:,test_inds(fold,:));
    % initialize weights
    v=randn(2,1);
    v0=randn(1);
    w=randn(2,2);
    w0=randn(2,1);
    dw = 0; dw0 = 0; dv = 0; dv0 = 0;

    % initialize data plot
    figure(1)
    clf
    subplot(121)
    hold on
    title('Layer 1 hyper-planes in data space')
    scatter(train_data(1,:), train_data(2,:), [], colors(train_labels+1,:))
    x1=[0 5];
    x2=-(w(1,1)*x1+w0(1))/w(2,1);
    h1=plot(x1,x2,'r--','LineWidth',2);
    x3=-(w(1,2)*x1+w0(2))/w(2,2);
    h2=plot(x1,x3,'r-.','LineWidth',2);
    axis image, axis([0 5 -2 3])

    subplot(122)
    hold on
    title('Layer 2 hyper-plane in layer-1 space')
    y=sigmoid(w'*train_data+w0*ones(1,K-(K/kfolds)));
    hy = scatter(y(1,:), y(2,:), [], colors(train_labels+1,:));
    y1=[-0.1 1.1];
    y2=-(v(1)*y1+v0)/v(2);
    h3=plot(y1,y2, 'r--', 'LineWidth',2);
    axis image, axis([-0.1 1.1 -0.1 1.1])
    
    E = zeros(num_trials,1);

    % loop over trials
    for t=1:num_trials

        % compute y layer
        y = sigmoid(w' * train_data + w0 * ones(1, K-(K/kfolds)));
        % compute output (z) layer
        output = sigmoid(v' * y + v0);

        % compute error
        delta = train_labels - output;
        E(t)  = delta * delta';
        % compute delta_z
        dsig    = output .* (1 - output);
        delta_z = delta .* dsig;
        % compute delta_y
        dsig    = y .* (1 - y);
        delta_y = dsig .* (v * delta_z);

        % accumulate dw
        dw  = eta * train_data * delta_y' + alpha * dw;
        dw0 = eta * sum(delta_y, 2) + alpha * dw0;
        % accumulate dv
        dv  = eta * y * delta_z' + alpha * dv ;
        dv0 = eta * sum(delta_z) + alpha * dv0;

        % update weights
        w  = w  + dw;
        w0 = w0 + dw0;
        v  = v  + dv;
        v0 = v0 + dv0;

        % update display of separating hyperplane
        x2=-(w(1,1)*x1+w0(1))/w(2,1);
        set(h1,'YData',x2)
        x3=-(w(1,2)*x1+w0(2))/w(2,2);
        set(h2,'YData',x3)

        y=sigmoid(w'*train_data+w0*ones(1,K-(K/kfolds)));
        set(hy, 'XData', y(1,:), 'YData', y(2,:))
        y2=-(v(1)*y1+v0)/v(2);
        set(h3,'YData',y2)

        drawnow
        
    end
    
    train_error(fold) = E(end);
    %---------------------CALCULATE TEST ERROR RATE-----------------------
    % note that this is a different loss function from the continuous sigmoid loss we calculate on the
    % training set, which performs better for back-propagation but is less
    % easily interpreted in terms of classification
    ti=0;
    test_preds = zeros(1,size(test_inds,2));
    for test_ind = test_inds(fold,:)
        ti = ti+1;
        test_preds(ti) = AO_discriminate(data(:,test_ind), v, v0, w, w0);
    end
    test_error(fold) = 1 - mean(test_preds == labels(test_inds(fold,:)));
    %---------------------end CALCULATE TEST ERROR RATE-----------------------
    fprintf('test error: %f \n', test_error(fold))


    figure(1)
    subplot(1,2,1)
    hold off
    subplot(1,2,2)
    hold off

    figure(2)
    subplot(1,5,fold)
    plot(E)
    title('Training error curve as a function of trial #')
    xlabel('Trial #')
    ylabel('Error')
end


%--------------------------PLOT TRAINING AND TEST ERROR ACROSS CV FOLDS---
figure(3)
subplot(1,2,1)
plot(train_error)
xlabel('Fold')
ylabel('Train error')
subplot(1,2,2)
plot(test_error)
xlabel('Fold')
ylabel('Test error rate')
title(sprintf('Mean test error: %f', mean(test_error)))
%--------------------------PLOT TRAINING AND TEST ERROR ACROSS CV FOLDS---

