function value = value_iteration(S, A, verbose)
    if nargin < 3
        verbose = true;
    end % if

    nstate = length(S);
    naction = length(A);
    npair = nstate * naction;
    
    fprintf('\nComputing state transitions...');
    tic();
    Snext = zeros(npair, const.DSTATE);
    for ipair = 1:npair
        [istate, iaction] = ind2sub([nstate, naction], ipair);
        Snext(ipair, :) = transition(S(istate, :), A(iaction));
    end % for ipair
    fprintf('done in %.2e\n', toc());
    
    fprintf('\nComputing next state indices for transition dictionary...');
    tic();
    ind_Snext = dsearchn(S, delaunayn(S), Snext);
    fprintf('done in %.2e\n', toc());
    
    fprintf('\nBeginning value iteration...\n');
    value = zeros(nstate, naction);
    residual = Inf;
    niter = 0;
    cputime = 0;
    while residual > const.VTOL
        tic();
        niter = niter + 1;
        residual = 0;
        
        for ipair = 1:npair
            [istate, iaction] = ind2sub([nstate, naction], ipair);
            [istate_next, iaction_next] = ...
                ind2sub([nstate, naction], ind_Snext(ipair));
            
            value_prev = value(istate, iaction);
            value_curr = ...
                reward(S(istate_next)) + const.GAMMA * value(istate_next, iaction_next);
            value(istate, iaction) = value_curr;
            
            residual = residual + (value_curr - value_prev)^2;
        end % for ipair
        
        iter_time = toc();
        if verbose
            fprintf('iter %d: residual = %.2e, time = %.2e\n', niter, residual, iter_time);
        end % if
        cputime = cputime + iter_time;
    end % while
    fprintf('\nValue iteration took %d iterations and %.2e sec\n', niter, cputime);
end % function