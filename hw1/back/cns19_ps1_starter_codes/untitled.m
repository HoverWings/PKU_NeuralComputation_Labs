%% Integration of Hodgkin--Huxley equations with Euler method
  clear; clf;
    
%actual code for problem
x = 0 : 0.5 : 15;
threshold = 30;
duration = 5;
for i = 0 : 0.5 : 15
  count = 0;
  for t = 1 : 1 : 10
    count = count + spikeFrequency(Integrate(i), threshold, duration);
  end
  y((2*i)+1) = count / 10;
end

plot(x, y);
xlabel("Current");
ylabel("Frequency");