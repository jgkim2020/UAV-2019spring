close all; clear all; clc;

% system dynamics
f_x = [0, 0, -9.81, 0, 0, 0].';
G_inv = [0, 0, 1, 0, 0, 0;
    0, 0, 0, 1, 0, 0;
    0, 0, 0, 0, 1, 0;
    0, 0, 0, 0, 0, 1;
    1, 0, 0, 0, 0, 0;
    0, 1, 0, 0, 0, 0];

% parameters (from Lee et. al, IJCAS 2009)
k = [1, 1, 0.7, 5, 5, 10]; % error gain
c = [5, 5, 5, 1, 1, 1]; % sliding mode gain
k_phi = 5; k_theta = 5;

% variables
x = [0, 1, 0, 0, 0, 0].'; % state (x, y, z, roll, pitch, yaw)
x_dot = [0, 0, 0, 0, 0, 0].';
x_2dot = [0, 0, 0, 0, 0, 0].';
x_d = [0, 1, 0, 0, 0, 0].'; % desired state
x_d_dot = [pi/2, 0, 1/2, 0, 0, pi/2].';
x_d_2dot = [0, -(pi/2)^2, 0, 0, 0, 0].';
phi_d = 0; theta_d = 0;
phi_d_dot = 0; theta_d_dot = 0;
s = [0, 0, 0, 0, 0, 0].'; % sliding surface
err = [0, 0, 0, 0, 0, 0].'; % error (x -x_d)
err_dot = [0, 0, 0, 0, 0, 0].';
err_x_ = 0; err_y_ = 0; % yaw corrected (body centric) error
nu_hat = [0, 0, 0, 0, 0, 0].';

% save for later
t_save = [];
x_save = [];
x_d_save = [];
U_save = [];
nu_hat_save = [];

%% run

dt = 0.001;
for t = 0:dt:8
    % control law
    U = G_inv*(-f_x + nu_hat + x_d_2dot - diag(k)*err_dot - diag(c)*sign(s));
    % estimator update law
    nu_hat_dot = -[err_dot(1) + k(1)*err(1), err_dot(2) + k(2)*err(2), 0, 0, 0, 0].';
    
    %%% save for later
    t_save = [t_save; t];
    x_save = [x_save; x.'];
    x_d_save = [x_d_save; x_d.'];
    U_save = [U_save; U.'];
    nu_hat_save = [nu_hat_save; nu_hat.'];
    
    %%% run system for dt and update state
    x_2dot(1) = U(1)*(cos(x(4))*sin(x(5))*cos(x(6)) + sin(x(4))*sin(x(6)));
    x_2dot(2) = U(1)*(cos(x(4))*sin(x(5))*sin(x(6)) - sin(x(4))*cos(x(6)));
    x_2dot(3) = U(1)*(cos(x(4))*cos(x(5))) - 9.81;
    x_2dot(4) = U(2);
    x_2dot(5) = U(3);
    x_2dot(6) = U(4);
    x_dot_old = x_dot; % t
    x_dot = x_dot + x_2dot*dt; % t + dt
    x = x + (x_dot_old + x_dot)/2*dt; % t + dt
    nu_hat = nu_hat + nu_hat_dot*dt; % t + dt
    
    %%% update other variables
    err_old = err; % t
    err(1) = x(1) - sin(pi/2*(t+dt)); % t + dt
    err(2) = x(2) - cos(pi/2*(t+dt)); % t + dt
    err_x_old_ = err_x_; err_y_old_ = err_y_; % t
    err_x_ = cos(x(6))*err(1) + sin(x(6))*err(2); % t + dt
    err_y_ = -sin(x(6))*err(1) + cos(x(6))*err(2); % t + dt
    err_x_dot_ = (err_x_ - err_x_old_)/dt; % t + dt
    err_y_dot_ = (err_y_ - err_y_old_)/dt; % t + dt
    % yaw corrected (body centric) phi_d & theta_d
    phi_d_old = phi_d; % t
    theta_d_old = theta_d; % t
    phi_d = err_y_dot_ + k_phi*err_y_; % t + dt
    theta_d = -(err_x_dot_ + k_theta*err_x_); % t + dt
    phi_d_dot_old = phi_d_dot; % t
    theta_d_dot_old = theta_d_dot; % t
    phi_d_dot = (phi_d - phi_d_old)/dt; % t + dt
    theta_d_dot = (theta_d - theta_d_old)/dt; % t + dt
    phi_d_2dot = (phi_d_dot - phi_d_dot_old)/dt; % t + dt
    theta_d_2dot = (theta_d_dot - theta_d_dot_old)/dt; % t + dt
    % desired trajectory @ t + dt
    x_d = [sin(pi/2*(t+dt)), cos(pi/2*(t+dt)), (t+dt)/2, phi_d, theta_d, pi/2*(t+dt)].';
    x_d_dot = [pi/2*cos(pi/2*(t+dt)), -pi/2*sin(pi/2*(t+dt)), 1/2, phi_d_dot, theta_d_dot, pi/2].';
    x_d_2dot = [-(pi/2)^2*sin(pi/2*(t+dt)), -(pi/2)^2*cos(pi/2*(t+dt)), 0, phi_d_2dot, theta_d_2dot, 0].';
    % error @ t + dt
    err = x - x_d;
    err_dot = (err - err_old)/dt;
    % sliding surface @ t + dt
    s = err_dot + diag(k)*err;
    
end

%% plot
close all;
figure('name', 'x, y');
plot(t_save, x_save(:,1));
hold on;
plot(t_save, x_save(:,2));
xlabel('time (sec)');
ylabel('x(t), y(t) (meter)');
legend('x', 'y');
figure('name', 'z');
plot(t_save, x_save(:,3));
xlabel('time (sec)');
ylabel('z(t) (meter)');
figure('name', 'yaw angle');
plot(t_save, x_save(:,6));
xlabel('time (sec)');
ylabel('\psi(t) (rad)');
figure('name', 'roll/pitch angle');
plot(t_save, x_save(:,4));
hold on;
plot(t_save, x_save(:,5));
xlabel('time (sec)');
ylabel('\phi(t), \theta(t) (rad)');
legend('\phi', '\theta');

figure('name', 'trajectory')
plot3(x_save(:,1), x_save(:,2), x_save(:,3));
hold on;
plot3(x_d_save(:,1), x_d_save(:,2), x_d_save(:,3), '--k');
xlabel('x (meter)');
ylabel('y (meter)');
zlabel('z (meter)');
legend('actual', 'desired');