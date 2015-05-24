function next_state = transition(state, action)
    next_state = zeros(2, 1);
    next_state(2) = clip(state(2) + 0.001 * action - 0.0025 * cos(3 * state(1)), const.VMIN, const.VMAX);
    next_state(1) = clip(state(1) + next_state(2), const.XMIN, const.XMAX);
end % function


function val_clipped = clip(val, minval, maxval)
    val_clipped = max(minval, min(maxval, val));
end % function