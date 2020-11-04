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

%%  The initial iterate:  a guess at the solution
x_0 = zeros(n,1);

%%  OPTIONAL:  give some extra instructions to FASTA using the 'opts' struct
opts = [];
opts.tol = 1e-8;  % Use super strict tolerance
opts.recordObjective = true; %  Record the objective function so we can plot it
opts.verbose=2;
opts.stringHeader='    ';      % Append a tab to all text output from FISTA.  This option makes formatting look a bit nicer. 

%%  Call fasta_solver
% Default behavior: adaptive stepsizes
[sol, outs_adapt] = fasta_sparseLeastSquares(A,A',b,gamma,x_0, opts);