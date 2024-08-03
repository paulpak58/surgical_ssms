function b = nanstd(a, arg)
    if nargin > 1
        dim = arg;
        b = std(a, dim, 'omitnan');
    else
        b = std(a, 'omitnan');
    end
    
end
