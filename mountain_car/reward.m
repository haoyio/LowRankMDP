function r = reward(state)
    if state(1) == const.XMAX
        r = const.TERM_REWARD;
    else
        r = const.NONTERM_REWARD;
    end % if
end % function