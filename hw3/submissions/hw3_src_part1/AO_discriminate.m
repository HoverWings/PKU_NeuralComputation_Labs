function [ out ] = AO_discriminate(x, v, v0, w, w0)

    sigmoid  = @(x) 1 ./ (1 + exp(-x));
    y = sigmoid(w' * x + w0);
    result = sigmoid(v' * y + v0);
    
    if result > 0.5
        out = 1;
    else
        out = 0;
    end

end

