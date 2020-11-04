% clear vars
clear;

s = RandStream.create('mt19937ar','seed',0);
RandStream.setGlobalStream(s);

m = 500;       % number of examples
n = 2500;      % number of features

x0 = sprandn(n,1,0.05);
A = randn(m,n);
A = A*spdiags(1./sqrt(sum(A.^2))',0,n,n); % normalize columns
v = sqrt(0.001)*randn(m,1);
b = A*x0 + v;

fprintf('solving instance with %d examples, %d variables\n', m, n);
fprintf('nnz(x0) = %d; signal-to-noise ratio: %.2f\n', nnz(x0), norm(A*x0)^2/norm(v)^2);

% gamma_max = norm(A'*b,'inf');
gamma = 0.3;

% cached computations for all methods
AtA = A'*A;
Atb = A'*b;

tic

% cvx implementation
cvx_begin quiet
    cvx_precision high
    variable x(n)
    minimize(0.5*sum_square(A*x - b) + gamma*norm(x,1))
cvx_end

h.x_cvx = x;
h.p_cvx = cvx_optval;
h.cvx_toc = toc;