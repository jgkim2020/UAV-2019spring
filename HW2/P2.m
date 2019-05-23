close all; clear all; clc;

% parmeters
global Ts
Ts = 0.1;
N = 20;
umax = 0.5;
max_iter = 10; % maximum iteration
alpha = 1e-1; % step size
epsilon = 1e-6; % finite difference step size

% variables
x = [0, 3, 0];
u = 0;

% receding horizon
x_horizon = zeros(N,3); % x_{1:N}
u_horizon = zeros(N,1); % u_{0:N-1}
lambda_horizon = zeros(N,3); % lambda_{1:N}
dHdu_horizon = zeros(N,1); % dH/du_{0:N-1}

% save for later
t_history = [];
x_history = [];
u_history = [];
J_history = [];

%% run

t_MPC = 0; % next scheduled MPC
dt = 0.01;
for t = 0:dt:15
    %%% run MPC every Ts
    if t_MPC <= t
        x0 = x;
        max_dHdu = Inf;
        iter_count = 0;
        while max_dHdu > 0.01
            if iter_count >= max_iter
                break;
            end
            
            % forward pass (u_horizon -> x_horizon)
            xk = x0;
            for k = 1:N
                x_horizon(k,1) = xk(1) + Ts*cos(xk(3));
                x_horizon(k,2) = xk(2) + Ts*sin(xk(3));
                x_horizon(k,3) = xk(3) + Ts*u_horizon(k);
                xk = x_horizon(k,:);
            end
            % backward pass (x,u_horizon -> lamda_horizon)
            lambda_horizon(N,:) = [0,+ 2*x_horizon(N,2), 0];
            lambdak = lambda_horizon(N,:);
            for k = N-1:-1:1
                xk = x_horizon(k,:);
                uk = u_horizon(k+1,:);
                Hk = H_val(xk, uk, lambdak)*[1, 1, 1];
                Hk_dx1 = H_val(xk + epsilon*[1, 0, 0], uk, lambdak);
                Hk_dx2 = H_val(xk + epsilon*[0, 1, 0], uk, lambdak);
                Hk_dx3 = H_val(xk + epsilon*[0, 0, 1], uk, lambdak);
                Hk_dx = [Hk_dx1, Hk_dx2, Hk_dx3];
                lambdak = (Hk_dx - Hk)/epsilon;
                lambda_horizon(k,:) = lambdak;
            end
            
            % compute dHdu (x,u,lambda_horizon -> dHdu_horizon)
            for k = 1:N
                if k == 1; xk = x0; else; xk = x_horizon(k-1,:); end
                uk = u_horizon(k,:);
                lambdak = lambda_horizon(k,:);
                Hk = H_val(xk, uk, lambdak);
                Hk_du = H_val(xk, uk + epsilon, lambdak);
                dHdu_horizon(k) = (Hk_du - Hk)/epsilon;
            end
            % update u
            u_horizon = u_horizon - alpha*dHdu_horizon;
            u_horizon = max(-umax, min(u_horizon, umax));
            
            max_dHdu = max(abs(dHdu_horizon));
            iter_count = iter_count + 1;
        end
        
        u = u_horizon(1);
%         u_horizon(1:N-1) = u_horizon(2:N); % shift control horizon
        u_horizon = zeros(N,1);
        u_horizon(N) = 0;
        t_MPC = t_MPC + Ts; % schedule next MPC
    end
    
    %%% save for later
    t_history = [t_history; t];
    x_history = [x_history; x];
    u_history = [u_history; u];
    
    %%% run system for dt
    x(1) = x(1) + cos(x(3))*dt;
    x(2) = x(2) + sin(x(3))*dt;
    x(3) = x(3) + u*dt;
    
end

%% plot
close all;

% state
figure('name', 'y');
plot(t_history, x_history(:,2));
xlabel('time (sec)');
ylabel('y(t) (meter)');
% control
figure('name', 'u');
plot(t_history, u_history);
xlabel('time (sec)');
ylabel('u(t)');
% trajectory
figure('name', 'trajectory');
plot(x_history(:,1), x_history(:,2));
xlabel('x(t) (meter)');
ylabel('y(t) (meter)');
axis equal;

%% functions
function H = H_val(x_k, u_k, lambda_kplus1)
    global Ts
    f_xu = [x_k(1) + Ts*cos(x_k(3)), x_k(2) + Ts*sin(x_k(3)), x_k(3) + Ts*u_k];
    H = x_k(2)^2 + dot(lambda_kplus1, f_xu);
end
