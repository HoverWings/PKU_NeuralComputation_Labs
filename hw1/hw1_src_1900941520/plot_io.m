%% This routine simulates the hh model (hh_func) and 
%% 
  clear; clf;
    
figure;
%actual code for problem
x = 0 : 0.5 : 15;
threshold = 30;
duration = 4;
count=0;
for i = 0 : 0.5 : 15
    count = 0;
    for t = 1 : 1 : 10
    disp(i);
%     disp('\n');
%     disp(t);
%     count=count+1;
    count = count + spikeFrequency(hh_func(i, 0.0001), threshold, duration);
    % the second parameter of hh_func is sigma that controls the noise
    % level. your task is to try different sigma, to see what happen to the
    % I/O curve. Right now, it is set to be 0, no noise. 
    end
    y((2*i)+1) = count / 10;
%     plot(x, y);
end

plot(x, y);
xlabel("Current");
ylabel("Frequency");

