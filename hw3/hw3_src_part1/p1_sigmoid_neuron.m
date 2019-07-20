% lab2s.m - single neuron learning
% you must first load in apples and oranges data
set(0, 'DefaultFigureWindowStyle', 'docked')

load apples;
load oranges;
load apples2;
load oranges2;

% substituting apples, apples2, oranges, and oranges2

fruitA = apples;
fruitB = apples2;

% initialize data array
data=[fruitA fruitB];
[N K]=size(data);

% initialize teacher
teacher=[ones(1,K/2) -zeros(1,K/2)];

% learning rate
eta= 0.5;

% number of trials - you may want to make this smaller or larger
num_trials= 250;

% initialize weights
w=randn(2,1);
w0=randn(1);

% initialize data plot
figure(1); clf
plot(fruitA(1,:),fruitA(2,:),'b+', fruitB(1,:), fruitB(2,:),'ro')
hold on
x1=0:4;
x2=-(w(1)*x1+w0)/w(2);
axis([0 4 -1 3])
h=plot(x1,x2);

E = zeros(1, num_trials);

sigmoid  = @(x) 1 ./ (1 + exp(-x));

% loop over trials
for t=1:num_trials
      
    % compute neuron output
    u = w' * data + w0;
    y = sigmoid(u);
      
    % compute delta
    delta = teacher - y;
    
    % compute error
    error = delta * delta';
    
    % compute dw
    dsig = y .* (1 - y);
    dw   = eta * data * (delta .* dsig)';
    dw0  = eta * dsig * delta';
   
    % update weights
    w  = w  + dw;
    w0 = w0 + dw0;
   
    % save E for this trial
    E(t) = sum(error);
   
    % update display of separating hyperplane
    x2=-(w(1)*x1+w0)/w(2);
    set(h,'YData',x2)
    drawnow
   
end

hold off

figure(2)
plot(E)

