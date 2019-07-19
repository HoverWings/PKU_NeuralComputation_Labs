% copied from hh.m and modified slightly, made a function for convenience (for parts 2b,2c,2e)
% this code generates 5 s worth of spikes. 5000  ms at 0.01 ms step
% it takes sigma in as a variable -- sigma randn generates the noise and add to the x, which is the membrane potential.
% you can do y = hh_func(15, 0.000), and then plot(y(1,1:5000) to plot the first seconds of spikes. 

function [y_plot] = hh_func( current, sigma )
    %% Setting parameters
    % Maximal conductances (in units of mS/cm^2); 1=K, 2=Na, 3=R
    g(1)=36; g(2)=120; g(3)=0.3;
    % Battery voltage ( in mV); 1=n, 2=m, 3=h
    E(1)=-12; E(2)=115; E(3)=10.613;
    % Initialization of some variables
    I_ext=0; V=-10; x=zeros(1,3); x(3)=1; t_rec=0;
    % Time step for integration
    dt=0.01;
    %     sigma=1;
%     k=1.0;
    %% Integration with Euler method
    for t=-30:dt:5000
        I_ext = current;
        if t>0; I_ext = current; end%+ sigma * randn(); end % + sigma * randn; end
        if t==40; I_ext=0; end   % turns external current off at t=40
    % alpha functions used by Hodgkin-and Huxley
    Alpha(1)=(10-V)/(100*(exp((10-V)/10)-1));
    Alpha(2)=(25-V)/(10*(exp((25-V)/10)-1));
    Alpha(3)=0.07*exp(-V/20);
    % beta functions used by Hodgkin-and Huxley
    Beta(1)=0.125*exp(-V/80);
    Beta(2)=4*exp(-V/18);
    Beta(3)=1/(exp((30-V)/10)+1);
    % tau_x and x_0 (x=1,2,3) are defined with alpha and beta
    tau=1./(Alpha+Beta);
    x_0=Alpha.*tau;
    % leaky integration with Euler method
    x_s=(1-dt./tau).*x+dt./tau.*x_0;
    x_n= sigma * randn;
    x= x_s+x_n;
%     x=(1-dt./tau).*x+dt./tau.*x_0 ;
    % calculate actual conductances g with given n, m, h
     gnmh(1)=g(1)*x(1)^4;
     gnmh(2)=g(2)*x(2)^3*x(3);
     gnmh(3)=g(3);
     
     gnmh_s(1)=g(1)*x_s(1)^4;
     gnmh_s(2)=g(2)*x_s(2)^3*x_s(3);
     gnmh_s(3)=g(3);
    % Ohm's law
     I=gnmh.*(V-E);
     I_s=gnmh_s.*(V-E);
    % update voltage of membrane
     V=V+dt*(I_ext-sum(I));
     V_s=V+dt*(I_ext-sum(I_s));
     
     dv=V_s-V;
%      a=x_s/x_n;
%      dv/a;
%      disp(a)
    % record some variables for plotting after equilibration
     if t>=0;
          t_rec=t_rec+1;
          x_plot(t_rec)=t;
          snr_plot(t_rec)=dv;
          y_plot(t_rec)=V;
     end
    end  % time loop
    % comment the following two plot lines if you are running  plot_io function

    plot(x_plot(100000:120000),snr_plot(100000:120000)); xlabel('Time'); ylabel('dv');
    title("dv with I_{ext} = 14 nA")
    print(gcf,'-dpng','abc.png');
    plot(x_plot(100000:120000),y_plot(100000:120000)); xlabel('Time'); ylabel('Voltage');
    title("Spiking activity with I_{ext} = 14 nA")
    print(gcf,'-dpng','abc1.png')
    end