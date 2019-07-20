% lab2m.m - multilayer network
% you must first load in apples and oranges data

load apples;
load oranges;
load apples2;
load oranges2;

sigmoid = @(x) 1 ./ (1 + exp(-x));

% initialize data array
%data=[apples oranges apples2 oranges2];
%data=[apples oranges2 oranges apples2];
%data=[apples oranges oranges2 apples2];
data=[apples apples2 oranges oranges2];
[N K]=size(data);

% initialize labels which will serve as our teaching signal
labels=[ones(1,K/2) zeros(1,K/2)];
half1=1:K/2;
half2=K/2+half1;

% learning rate
eta=.1;
alpha=.5;  % momentum term

% number of trials - you may want to make this longer
num_trials=1000;

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
plot(data(1,half1),data(2,half1),'b+',data(1,half2),data(2,half2), 'ro')
hold on
x1=[0 5];
x2=-(w(1,1)*x1+w0(1))/w(2,1);
h1=plot(x1,x2,'y--','LineWidth',2);
x3=-(w(1,2)*x1+w0(2))/w(2,2);
h2=plot(x1,x3,'y-.','LineWidth',2);
axis image, axis([0 5 -2 3])

subplot(122)
datay=sigmoid(w'*data+w0*ones(1,K));
hy(1)=plot(datay(1,half1),datay(2,half1),'b+');
hold on
hy(2)=plot(datay(1,half2),datay(2,half2),'ro');
y1=[-0.1 1.1];
y2=-(v(1)*y1+v0)/v(2);
h3=plot(y1,y2,'LineWidth',2);
axis image, axis([-0.1 1.1 -0.1 1.1])

% loop over trials
for t=1:num_trials
   
    % compute y layer
    y = sigmoid(w' * data + w0 * ones(1, K));
    
    % compute z layer
    z = sigmoid(v' * y + v0);
    
    % compute error
    delta = labels - z;
    E(t)  = delta * delta';
    
    % compute delta_z
    dsig    = z .* (1 - z);
    delta_z = delta .* dsig;
    
    % compute delta_y
    dsig    = y .* (1 - y);
    delta_y = dsig .* (v * delta_z);
    
    % accumulate dw
    dw  = eta * data * delta_y' + alpha * dw;
    dw0 = eta * sum(delta_y, 2) + alpha * dw0;
    
    % accumulate dv
    dv  = eta * y * delta_z' + alpha * dv ;
    dv0 = eta * sum(delta_z) + alpha * dv0;
    
    % update weights
    w  = w  + dw;
    w0 = w0 + dw0;
    v  = v  + dv;
    v0 = v0 + dv0;
   
    % save E for this trial
   
    % update display of separating hyperplane
    x2=-(w(1,1)*x1+w0(1))/w(2,1);
    set(h1,'YData',x2)
    x3=-(w(1,2)*x1+w0(2))/w(2,2);
    set(h2,'YData',x3)
   
    datay=sigmoid(w'*data+w0*ones(1,K));
    set(hy(1),'XData',datay(1,half1),'YData',datay(2,half1))
    set(hy(2),'XData',datay(1,half2),'YData',datay(2,half2))
    y2=-(v(1)*y1+v0)/v(2);
    set(h3,'YData',y2)
   
%     % update E plot
%     if t>1
%       set(hE,'XData',[t-1 t],'YData',E(t-1:t))
%     else
%       figure(2)
%       hE=plot(1,E(1),'EraseMode','none');
%       axis([1 num_trials 0 E(1)])
%     end
   
    drawnow
   
end

figure(1)
subplot(121)
title('Layer 1 hyper-planes in data space')
hold off
subplot(122)
title('Layer 2 hyper-plane in layer-1 space')
hold off

figure(2)
plot(E)
title('Training error curve as a function of trial #')
xlabel('Trial #')
ylabel('Error')

% print
%set(0,'defaulttextinterpreter','none');
%printfig(1,'q3_1',7);