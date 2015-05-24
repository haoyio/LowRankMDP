classdef const
    properties (Constant)
        % mdp
        N = 100;                % problem size
        GAMMA = 0.95;           % discount parameter
        
        % state
        DSTATE = 2;             % state dimension/number of state variables
        XMIN = -1.2;            % minimum position
        XMAX = 0.5;             % maximum position
        VMIN = -0.07;           % minimum speed
        VMAX = 0.07;            % maximum speed
        
        % action
        AMIN = -1.0;            % minimum acceleration
        AMAX = 1.0;             % maximum acceleration
        
        % reward
        TERM_REWARD = 10;       % terminal state reward
        NONTERM_REWARD = -1;    % non-terminal state reward
        
        % value iteration
        VTOL = 1e-6;    % convergence tolerance
    end % properties
end % classdef