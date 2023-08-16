% This script refers to our magazine submitted to the IEEE Network whose 
% title is Semantic-Functional Communications in Cyber-Physical Systems.

% In this MATLAB script, a set of drones performing a random walk with the 
% objective of monitoring a certain area. Each drone is assigned a portion 
% of the total area. The drones are programmed to return to the central 
% position of their coverage area when they receive a return command 
% indicating that they are outside of their specific region. 

% For more theoretical details, we advise you to read the manuscript 
% Semantic-Functional Communications in Cyber-Physical Systems.

%% Inputs / Adjustable variables for the simulation

% square breadths of boundarie. Defines the coverage area of a drone.
a = 20;
b = 20;

% square breadths of centre. Defines the size of central position.
ac = 1;
bc = 1;

% error probability
% A semantic-functional communication is adopted for signaling events. 
% Such a method makes it possible for an event to always be received by 
% the decision maker at the cost of false alarms. For more details see
% A novel semantic-functional approach for multiuser event-trigger 
% communication at https://doi.org/10.48550/arXiv.2204.03223 and 
% Enabling Semantic-Functional Communications for Multiuser Event 
% Transmissions via Wireless Power Transfer at https://doi.org/10.3390/s23052707

Pe_fa = 2e-3;   % false alarm
Pe_fp = 0;      % does not note event

% semantic-functional state. Use 1, 2 or 3.  
s = 1;

% For S1=0, i.e., no system state is considered, s=1. 
% This is an extreme case where the system is not monitored and there is no
% communication from sensors to the decision-maker. However, the 
% decision-maker continues to act on the drones by randomly sending a 
% return command to each drone.
max_timeout = 5; % seconds
min_timeout = 2; % seconds
tm = 0; % turn on timeout

% drone features
max_velocity = 10; % [m/s], maximum speed of the dorne in meters per second
a_rate = 1.1; % speed reduction rate so that the drone can smoothly stop in the central area.
coverage = 3; % coverage circle radius. Use > max_velocity*signaling_time. 
% It refers to the area that the drone can cover when it is flying.
N_dones = 9; % number of drones. 

cont_delay = 1; % dalay in steps
signaling_time = 1e-1; % time of simulation step in seconds

K_steps = 1e6; % the length of the random walks.
N_sim = 1e2; % number of runs of the random walk.

% tests which semantic-functional state has been defined and adjusts the 
% tm and rand_delay parameters.
% rand_delay is a random delay caused by TDMA technique. 
try
    if s==1
        tm = 1;
        rand_delay = 0;
    elseif s==2
        tm = 0;
        rand_delay = 0;
    elseif s==3
        tm = 0;
        rand_delay = 1;
    end
catch
    s=0;
end

%% grid of covarage area. 
% Defines a grid with the partitions of the area that the drone can cover 
% at each step of the simulation.
xgrid = [];
for v = -a/2:coverage:0
xgrid = [xgrid, v];
end
xgrid = [xgrid flip(abs(xgrid(xgrid~=0)))];
% xgrid_l = length(xgrid);

ygrid = [];
for v = -b/2:coverage:0
ygrid = [ygrid, v];
end
ygrid = [ygrid flip(abs(ygrid(ygrid~=0)))];
% ygrid_l = length(ygrid);

cov_area = zeros(length(ygrid)-1,length(xgrid)-1);
t_cov = 0;

%% Random walk. Run the simulation

% Prepare some variables to monitor the simulation.
res_event = zeros(1,N_sim);
res_t_cov = zeros(1,N_sim);
kpi = zeros(1,N_sim);
cont = 0;
h = waitbar(0,'no way...');

for sim = 1:N_sim
    
    waitbar( sim/N_sim, h)
    x_t = zeros(1,K_steps); % drone position on the x axis
    y_t = zeros(1,K_steps); % % drone position on the y axis
    returning = 0; % return route activated
    event = zeros(1,K_steps);
    % cont = 0;
    timeout = randi([min_timeout, max_timeout]);
    % Run a K_steps step random walk
    for t = 1:K_steps
        % Checks if drone should stop returning and start a new random walk
        if abs(x_t(t)) < ac/2 && abs(y_t(t)) < bc/2
            returning = 0;
        end
        
        % Updates the drone's coverage record
        if abs(x_t(t)) < a/2 && abs(y_t(t)) < b/2
            c_x=2;
            while x_t(t)>xgrid(c_x)
                c_x = c_x + 1;
            end
            c_x = c_x-1;
            c_y=2;
            while y_t(t)>ygrid(c_y)
                c_y = c_y + 1;
            end
            c_y = c_y-1;

            cov_area(c_y,c_x) = 1;
            if all(cov_area,'all')
                t_cov(end) = t;
                t_cov = [t_cov 0];
                cov_area = zeros(length(ygrid)-1,length(xgrid)-1);
            end
        end
        
        % For s=1, initiate return if a timeout occurs
        if cont > 0 % return command delay 
            cont = cont - 1;
            if cont <= 0
                returning = 1;
            end
        end
        if tm == 1 % timeout
            if timeout <= 0
                cont = 1;
                timeout = randi([min_timeout, max_timeout]);
            else
                timeout = timeout - signaling_time;
            end
        else
            if (abs(x_t(t))>a/2 || abs(y_t(t))>b/2) && (abs(x_t(t-1))<=a/2 && abs(y_t(t-1))<=b/2)
                event(t) = 1;
                if rand >= Pe_fp
                    if rand_delay == 1
                        cont = randi([3,N_dones*3]) + cont_delay;
                    else
                        cont = cont_delay;
                    end
                    
                end
            end
            if (rand < Pe_fa)
                returning = 1;
            end
        end
        
        % adjusts the direction of drone to return to the center.
        if returning == 1
            if abs(x_t(t)) > ac/2
                if abs(x_t(t)/a_rate) > max_velocity
                    Vx = -max_velocity*sign(x_t(t));
                else
                    Vx = -x_t(t)/a_rate;
                end
            else
                Vx=0;
            end

            if abs(y_t(t)) > bc/2
                if abs(y_t(t)/a_rate) > max_velocity
                    Vy = -max_velocity*sign(y_t(t));
                else
                    Vy = -y_t(t)/a_rate;
                end
            else
                Vy=0;
            end
        else
    %         Vx = sqrt(var_velocity)*randn;
    %         Vy = sqrt(var_velocity)*randn;

    %         Vx = 2*max_velocity*rand - max_velocity;
    %         Vy = 2*max_velocity*rand - max_velocity;

            ang = 2*pi*rand;
            Vx = max_velocity*cos(ang);
            Vy = max_velocity*sin(ang);
        end
        
        % updates drone position
        x_t(t+1) = x_t(t) + Vx*signaling_time; 
        y_t(t+1) = y_t(t) + Vy*signaling_time;
        
    end
    res_event(sim) = mean(event);
    t_cov = t_cov(t_cov~=0);
    for t = length(t_cov):-1:2
        t_cov(t) = t_cov(t) - t_cov(t-1);
    end
    res_t_cov(sim) = mean(t_cov);
    t_cov = 0;
    kpi(sim) = sum((abs(x_t)>a/2) | (abs(y_t)>b/2))/K_steps;
end

close( h );

%% Results

figure();
% subplot(2,3,s);
plot(x_t(1:2e3), y_t(1:2e3));
hold on;
% grid on;
line([-a/2 a/2],[-b/2 -b/2],'color','r','LineStyle','--','LineWidth',3);
line([-a/2 a/2],[b/2 b/2],'color','r','LineStyle','--','LineWidth',3);
line([-a/2 -a/2],[-b/2 b/2],'color','r','LineStyle','--','LineWidth',3);
line([a/2 a/2],[-b/2 b/2],'color','r','LineStyle','--','LineWidth',3);
% Enlarge figure to full screen.
set(gcf, 'Units', 'Normalized', 'Outerposition', [0, 0.05, 1, 0.95]);
axis([-20 20 -20 20]);
axis square

% subplot(2,3,s+3);
figure();
plot(x_t, y_t);
hold on;
% grid on;
line([-a/2 a/2],[-b/2 -b/2],'color','r','LineStyle','--','LineWidth',3);
line([-a/2 a/2],[b/2 b/2],'color','r','LineStyle','--','LineWidth',3);
line([-a/2 -a/2],[-b/2 b/2],'color','r','LineStyle','--','LineWidth',3);
line([a/2 a/2],[-b/2 b/2],'color','r','LineStyle','--','LineWidth',3);
% Enlarge figure to full screen.
set(gcf, 'Units', 'Normalized', 'Outerposition', [0, 0.05, 1, 0.95]);
axis([-20 20 -20 20]);
axis square
% 
% figure();
% r_t = sqrt(x_t.^2 + y_t.^2);
% histogram(r_t,'Normalization','pdf')
% 
% kpi = sum((abs(x_t)>a/2) | (abs(y_t)>b/2))/K_steps;
% 
fprintf('\n   Results for S_{%1d}  \n',s);
fprintf('perimeter violation: %2.5f%% \n \n', mean(kpi)*100);
fprintf('average time / optimal time: ');
disp(N_dones*mean(res_t_cov)/numel(cov_area));
fprintf('average transmissions per unit of time: ');
disp(mean(res_event)/signaling_time);
fprintf('average time to cover area given in hours: ');
disp(mean(res_t_cov/3600)*signaling_time);
fprintf('average time to cover area given given in minutes: ');
disp(mean(res_t_cov/60)*signaling_time);

