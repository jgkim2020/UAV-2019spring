close all; clear all; clc;

rng(1337); % set seed

N = 30; k = 10;
T = 50; dt = 0.02;
K = ones(N, N);
R = 0.1.*ones(N, N);
% initial state
xmin = -2; xmax = 2;
vmin = -0.5; vmax = 0.5;
xs = xmin + (xmax - xmin).*rand(N,2);
vs = vmin + (vmax - vmin).*rand(N,2);
state = [xs, vs];

%% (a) fully-connected network
t_history = [];
state_history = {};
com_history = [];
for t = 0:dt:T
    % control inputs
    state_dot = zeros(N,4);
    for i = 1:N
        xi_dot = state(i,3:4);
        vi_dot = zeros(1,2);
        for j = 1:N
            vi_dot = vi_dot + K(i,j)/N.*(state(j,3:4) - state(i,3:4));
        end
        state_dot(i,1:2) = xi_dot;
        state_dot(i,3:4) = vi_dot;
    end
    % save for later
    t_history = [t_history; t];
    state_history{end + 1} = state;
    com_history = [com_history; mean(state)];
    % run system for dt
    state = state + dt.*state_dot;
end

figure('name', 'trajectories');
hold on;
% plot individual trajectories
for i = 1:N
    pos = [];
    for t = 1:length(state_history)
        pos = [pos; state_history{t}(i,1:2)];
    end
    plot(pos(:,1),pos(:,2));
end
% plot COM trajectory
plot(com_history(:,1),com_history(:,2),'--k');
xlabel('x (meter)');
ylabel('y (meter)');

figure('name', 'COM velocity');
hold on;
plot(t_history, com_history(:,3));
plot(t_history, com_history(:,4));
legend('x', 'y');
xlabel('time (sec)');
ylabel('velocity (m/s)');

%% (b) locally-connected network
t_history = [];
state_history = {};
com_history = [];
for t = 0:dt:T
    % distance between agents
    dij = zeros(N,N);
    for i = 1:N
        for j = 1:N
            dij(i,j) = sqrt(sum((state(i,1:2) - state(j,1:2)).^2));
        end
    end
    % control inputs
    state_dot = zeros(N,4);
    for i = 1:N
        xi_dot = state(i,3:4);
        vi_dot = zeros(1,2);
        dist2js = zeros(N,1);
        [vals, idxs] = mink(dij(i,:),k+1);
        for j = idxs 
            vi_dot = vi_dot + K(i,j)/N.*(state(j,3:4) - state(i,3:4));
        end
        state_dot(i,1:2) = xi_dot;
        state_dot(i,3:4) = vi_dot;
    end
    % save for later
    t_history = [t_history; t];
    state_history{end + 1} = state;
    com_history = [com_history; mean(state)];
    % run system for dt
    state = state + dt.*state_dot;
end

figure('name', 'trajectories');
hold on;
% plot individual trajectories
for i = 1:N
    pos = [];
    for t = 1:length(state_history)
        pos = [pos; state_history{t}(i,1:2)];
    end
    plot(pos(:,1),pos(:,2));
end
% plot COM trajectory
plot(com_history(:,1),com_history(:,2),'--k');
xlabel('x (meter)');
ylabel('y (meter)');

figure('name', 'COM velocity');
hold on;
plot(t_history, com_history(:,3));
plot(t_history, com_history(:,4));
legend('x', 'y');
xlabel('time (sec)');
ylabel('velocity (m/s)');

%% (c) locally-connected network (with repulsive force)
t_history = [];
state_history = {};
com_history = [];
for t = 0:dt:T
    % distance between agents
    dij = zeros(N,N);
    for i = 1:N
        for j = 1:N
            dij(i,j) = sqrt(sum((state(i,1:2) - state(j,1:2)).^2));
        end
    end
    % control inputs
    state_dot = zeros(N,4);
    for i = 1:N
        xi_dot = state(i,3:4);
        vi_dot = zeros(1,2);
        dist2js = zeros(N,1);
        [vals, idxs] = mink(dij(i,:),k+1);
        for j = idxs 
            flocking_force = K(i,j)/N.*(state(j,3:4) - state(i,3:4));
            dist = sqrt(sum((state(i,1:2) - state(j,1:2)).^2));
            if dist < 1e-2
                repulsive_force = 0;
            else
                repulsive_force = 0.1/dist/(R(i,j) + dist)/N.*(state(i,1:2) - state(j,1:2));
            end
            vi_dot = vi_dot + flocking_force + repulsive_force;
        end
        state_dot(i,1:2) = xi_dot;
        state_dot(i,3:4) = vi_dot;
    end
    % save for later
    t_history = [t_history; t];
    state_history{end + 1} = state;
    com_history = [com_history; mean(state)];
    % run system for dt
    state = state + dt.*state_dot;
end

figure('name', 'trajectories');
hold on;
% plot individual trajectories
for i = 1:N
    pos = [];
    for t = 1:length(state_history)
        pos = [pos; state_history{t}(i,1:2)];
    end
    plot(pos(:,1),pos(:,2));
end
% plot COM trajectory
plot(com_history(:,1),com_history(:,2),'--k');
xlabel('x (meter)');
ylabel('y (meter)');

figure('name', 'COM velocity');
hold on;
plot(t_history, com_history(:,3));
plot(t_history, com_history(:,4));
legend('x', 'y');
xlabel('time (sec)');
ylabel('velocity (m/s)');