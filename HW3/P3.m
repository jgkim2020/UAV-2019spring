close all; clear all; clc;

rng(1337);

numrows = 4; numcols = 7;
p_fault = 0.0;
obstacles = [6, 9, 13, 16, 25];
goal = 7;
gamma = 0.99;
r_step = -0.04; r_goal = 1.0;
N_iter = 6;

N_states = numrows*numcols;
actions_label = {'R', 'U', 'L', 'D'};
actions = {[0,1], [-1,0], [0,-1], [1,0]};
N_actions = length(actions_label);

% find transition prob.
P_tr = zeros(N_states, N_actions, N_states); % (s,a,s')
for i = 1:numrows
    for j = 1:numcols
        state = numcols*(i-1) + j;
        if ~any(obstacles == state) % not an obstacle
            for a = 1:N_actions
                
                p_sa = zeros(1,N_states);
                for a_ = 1:N_actions
                    % state transition
                    ij_ = [i, j] + actions{a_};
                    state_ = numcols*(ij_(1)-1) + ij_(2);
                    if (ij_(1) < 1) || (ij_(1) > numrows) || (ij_(2) < 1) || (ij_(2) > numcols)
                        state_ = state; % out of map?
                    elseif any(obstacles == state_)
                        state_ = state; % is an obstacle?
                    else
                        state_ = state_; % none of the above
                    end
                    % assign probability
                    if a_ == a
                        p_sa(state_) = p_sa(state_) + 1 - 3*p_fault; 
                    else
                        p_sa(state_) = p_sa(state_) + p_fault;
                    end
                end
                % save to P_tr
                P_tr(state, a, :) = p_sa;
                
            end
        end
    end
end

% random policy
policy = randi([1,4], N_states, 1);
for s = 1:N_states
    if s == goal || any(obstacles == s)
        policy(s) = 0;
    end
end
% initial value function
values = zeros(N_states, 1);

%% policy iteration
values_history = [];
values_iter_history = [];
policy_history = [];
policy_diff_history = [];

done = false;
for iter = 1:N_iter
    % policy evaluation
    values_diff = Inf;
    values_iter = 0;
    while values_diff > 0.001
        values_new = zeros(size(values));
        for s = 1:N_states
            if s == goal || any(obstacles == s) % goal or obstacle
                values_new(s) = 0;
                continue;
            end
            a = policy(s);
            for s_ = 1:N_states
                if s_ == goal
                    reward = r_goal;
                else
                    reward = r_step;
                end
                values_new(s) = values_new(s) + P_tr(s,a,s_)*(reward + gamma*values(s_));
            end
        end
        values_iter = values_iter + 1;
        values_diff = max(abs(values_new - values));
        values = values_new;
    end
    values_iter_history = [values_iter_history, values_iter];
    values_history = [values_history, values];
    
    % policy improvement (greedy)
    policy_new = zeros(size(policy));
    for s = 1:N_states
        if s == goal || any(obstacles == s) % goal or obstacle
            policy_new(s) = 0;
            continue;
        end
        R = zeros(1, N_actions);
        for a = 1:N_actions
            for s_ = 1:N_states
                if s_ == goal
                    reward = r_goal;
                else
                    reward = r_step;
                end
                R(a) = R(a) + P_tr(s,a,s_)*(reward + gamma*values(s_));
            end
        end
        [val, idx] = maxk(R,1); % greedy
        policy_new(s) = idx;
    end
    policy_history = [policy_history, policy];
    policy_diff_history = [policy_diff_history, nnz(policy_new - policy)];
    policy = policy_new;
end

% plot
plot_values = zeros(numrows, numcols, N_iter);
plot_policy = strings(numrows, numcols, N_iter);
for iter = 1:N_iter
    for i = 1:numrows
        for j = 1:numcols
            state = numcols*(i-1) + j;
            plot_values(i,j,iter) = values_history(state,iter);
            if policy_history(state,iter) ~= 0
                plot_policy(i,j,iter) = actions_label(policy_history(state,iter));
            end
        end
    end
end