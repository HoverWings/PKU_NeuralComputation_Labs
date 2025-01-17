% % corrupt.m - corrupts a pattern by flipping bits
% %

% function Inew = corrupt(I,n)

% N=length(I);

% i=ceil(N*rand(n,1));
% % foo = randperm(N);
% % i = foo(1:n);

% Inew=I;
% Inew(i)=-I(i);
% end


% corrupt.m - corrupts a pattern by flipping bits
%

function Inew = corrupt(I,n)
    N=length(I);

    % i=ceil(N*rand(n,1));
    foo = randperm(N);
    i = foo(1:n);

    Inew=I;
    % Inew(i)=-I(i);
    Inew(i) = (randi(2, n, 1) - 1.5) .* 2
end
