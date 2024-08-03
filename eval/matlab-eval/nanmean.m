function b = nanmean(a, arg)
    if nargin > 1
        dim = arg;
        b = mean(a, dim, 'omitnan');
    else
        b = mean(a, 'omitnan');
    end
    
end