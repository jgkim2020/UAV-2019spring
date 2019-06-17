close all; clear all; clc;
addpath('./hungarian');

N = 8;
T = 25; dt = 0.01;
pgain = 1;
% initial state
xs = zeros(N,2);
xs(1,1) = -40; xs(2,1) = 40;
xs(3,1) = -30; xs(4,1) = 30;
xs(5,1) = -20; xs(6,1) = 20;
xs(7,1) = -10; xs(8,1) = 10;
vs = zeros(N,2);
% desired state
xdes = zeros(N,2);
xdes(1,1) = -20; xdes(2,1) = 20;
xdes(3,2) = -20; xdes(4,2) = 20;
xdes(5,1) = -14; xdes(5,2) = 14;
xdes(6,1) = 14; xdes(6,2) = 14;
xdes(7,1) = -14; xdes(7,2) = -14;
xdes(8,1) = 14; xdes(8,2) = -14;

%% least total distance
t_history = [];
state_history = {};
% cost(distance) matrix
dij = zeros(N,N);
for i = 1:N
    for j = 1:N
        dij(i,j) = sqrt(sum((xs(i,:) - xdes(j,:)).^2));
    end
end
% Hungarian algorithm
[I, J] = linear_sum_assignment(dij);
match = zeros(1,N);
for i = 1:N
    match(I(i)) = J(i);
end
% move to desired
for t=0:dt:T
    % control inputs
    xdots = zeros(N,2);
    for i = 1:N
        xdot = pgain*(xdes(match(i),:) - xs(i,:));
        xdots(i,:) = round(min(max(xdot,-2),2)); % in range -2, -1, 0, 1, 2
    end
    % save for later
    t_history = [t_history, t];
    state_history{end + 1} = xs;
    % run system for dt
    xs = xs + dt.*xdots;
end

% plot trajectories
figure('name', 'trajectories');
axis equal;
hold on;
for i = 1:N
    pos = [];
    for t = 1:length(state_history)
        pos = [pos; state_history{t}(i,:)];
    end
    plot(pos(:,1),pos(:,2));
end
viscircles([0,0], 20, 'Color', 'k', 'LineStyle', '--');