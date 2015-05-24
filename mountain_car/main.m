close all; clear; clc;

% initialize problem
x = linspace(const.XMIN, const.XMAX, const.N);
v = linspace(const.VMIN, const.VMAX, const.N);
X = repmat(x(:).', [length(v) 1]);
V = repmat(v(:), [1 length(x)]);
S = [X(:) V(:)];
A = linspace(const.AMIN, const.AMAX, const.N);

% perform value iteration
value = value_iteration(S, A);
save('value.mat', value);

imagesc(value);