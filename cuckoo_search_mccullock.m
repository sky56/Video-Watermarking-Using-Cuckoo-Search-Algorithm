% -----------------------------------------------------------------
% Cuckoo Search (CS) algorithm by Xin-She Yang and Suash Deb      %
% Programmed by Xin-She Yang at Cambridge University         a     %
% Programming dates: Nov 2008 to June 2009                        %
% Last revised: Dec  2009   (simplified version for demo only)    %
% -----------------------------------------------------------------
% Papers -- Citation Details:
% 1) X.-S. Yang, S. Deb, Cuckoo search via Levy flights,
% in: Proc. of World Congress on Nature & Biologically Inspired
% Computing (NaBIC 2009), December 2009, India,
% IEEE Publications, USA,  pp. 210-214 (2009).
% http://arxiv.org/PS_cache/arxiv/pdf/1003/1003.1594v1.pdf 
% 2) X.-S. Yang, S. Deb, Engineering optimization by cuckoo search,
% Int. J. Mathematical Modelling and Numerical Optimisation, 
% Vol. 1, No. 4, 330-343 (2010). 
% http://arxiv.org/PS_cache/arxiv/pdf/1005/1005.2908v2.pdf
% ----------------------------------------------------------------%
% This demo program only implements a standard version of         %
% Cuckoo Search (CS), as the Levy flights and generation of       %
% new solutions may use slightly different methods.               %
% The pseudo code was given sequentially (select a cuckoo etc),   %
% but the implementation here uses Matlab's vector capability,    %
% which results in neater/better codes and shorter running time.  % 
% This implementation is different and more efficient than the    %
% the demo code provided in the book by 
%    "Yang X. S., Nature-Inspired Metaheuristic Algoirthms,       % 
%     2nd Edition, Luniver Press, (2010).                 "       %
% --------------------------------------------------------------- %

% =============================================================== %
% Notes:                                                          %
% Different implementations may lead to slightly different        %
% behavour and/or results, but there is nothing wrong with it,    %
% as this is the nature of random walks and all metaheuristics.   %
% -----------------------------------------------------------------

% Additional Note: This version uses a fixed number of generation %
% (not a given tolerance) because many readers asked me to add    %
%  or implement this option.                               Thanks.%                          
function [fmin] = cuckoo_search_mccullock(n)
if nargin<1,
% Number of nests (or different solutions)
n=20;
end

% Discovery rate of alien eggs/solutions
pa=0.25;

%% Change this if you want to get better results
N_IterTotal=250;
%% Simple bounds of the search domain
% Lower bounds
nd=15; 
Lb=-5*ones(1,nd); 
% Upper bounds
Ub=5*ones(1,nd);

% Random initial solutions
for i=1:n,
nest(i,:)=Lb+(Ub-Lb).*rand(size(Lb));
end

% Get the current best
fitness=10^10*ones(n,1);
[fmin,bestnest,nest,fitness]=get_best_nest(nest,nest,fitness);

N_iter=0;
%% Starting iterations
for iter=1:N_IterTotal,
    % Generate new solutions (but keep the current best)
     new_nest=get_cuckoos(nest,bestnest,Lb,Ub);   
     [fnew,best,nest,fitness]=get_best_nest(nest,new_nest,fitness);
    % Update the counter
      N_iter=N_iter+n; 
    % Discovery and randomization
      new_nest=empty_nests(nest,Lb,Ub,pa) ;
    
    % Evaluate this set of solutions
      [fnew,best,nest,fitness]=get_best_nest(nest,new_nest,fitness);
    % Update the counter again
      N_iter=N_iter+n;
    % Find the best objective so far  
    if fnew<fmin,
        fmin=fnew;
        bestnest=best;
    end
end %% End of iterations

%% Post-optimization processing
%% Display all the nests
%disp(strcat('Total number of iterations=',num2str(N_iter)));
fmin
%bestnest

%% --------------- All subfunctions are list below ------------------
%% Get cuckoos by ramdom walk
function nest=get_cuckoos(nest,best,Lb,Ub)
% Levy flights
n=size(nest,1);
% Levy exponent and coefficient
% For details, see equation (2.21), Page 16 (chapter 2) of the book
% X. S. Yang, Nature-Inspired Metaheuristic Algorithms, 2nd Edition, Luniver Press, (2010).
beta=3/2;
delta=3/2;
alpha=2;
sigma=(gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function x = MuCulloch_levy_flight(n,alpha,sigma,beta,delta,type)

% gen_levy_flight - generate a Levy flight
%
%   x = gen_levy_flight(n,alpha,sigma,beta,delta,type);
%
%   n is the length
%   alpha is the exponent (alpha=2 for gaussian, alpha=1 for cauchian)
%   sigma is the standard deviation
%   beta and delta are symmetry parameter (for no drift, set to 0)
%   type is either 'isotropic' or 'axis'
%
%   Copyright (c) 2005 Gabriel Peyré


if nargin<2
    alpha = 1;
end
if nargin<3
    sigma = 1;
end
if nargin<4
    beta = 0;
end
if nargin<5
    delta = 0;
end
if nargin<6
    type = 'isotropic';
end

x = zeros(n,2);
if strcmp(type, 'isotropic')
    r = stabrnd(alpha, beta, sigma, delta, 1, n);
    theta = 2*pi*rand(1,n);
    x(:,1) = r.*cos(theta);
    x(:,2) = r.*sin(theta);
else
    x(:,1) = stabrnd(alpha, beta, c, delta, 1, n);
    x(:,2) = stabrnd(alpha, beta, c, delta, 1, n);
end
x = cumsum(x);


function [x] = stabrnd(alpha, beta, c, delta, m, n)

% STABRND.M
% Stable Random Number Generator (McCulloch 12/18/96)
%
%   x = stabrnd(alpha, beta, c, delta, m, n);
%
% Returns m x n matrix of iid stable random numbers with 
%   characteristic exponent alpha in [.1,2], skewness parameter
%   beta in [-1,1], scale c > 0, and location parameter delta.
% Based on the method of J.M. Chambers, C.L. Mallows and B.W.
%   Stuck, "A Method for Simulating Stable Random Variables," 
%   JASA 71 (1976): 340-4.  
% Encoded in MATLAB by J. Huston McCulloch, Ohio State
%   University Econ. Dept. (mcculloch.2@osu.edu).  This 12/18/96
%   version uses 2*m*n calls to RAND, and does not rely on 
%   the STATISTICS toolbox.
% The CMS method is applied in such a way that x will have the 
%   log characteristic function 
%        log E exp(ixt) = i*delta*t + psi(c*t), 
%   where
%     psi(t) = -abs(t)^alpha*(1-i*beta*sign(t)*tan(pi*alpha/2))
%                              for alpha ~= 1,
%            = -abs(t)*(1+i*beta*(2/pi)*sign(t)*log(abs(t))),
%                              for alpha = 1.
% With this parameterization, the stable cdf S(x; alpha, beta, 
%   c, delta) equals S((x-delta)/c; alpha, beta, 1, 0).  See my 
%   "On the parametrization of the afocal stable distributions,"
%   _Bull. London Math. Soc._ 28 (1996): 651-55, for details.
% When alpha = 2, the distribution is Gaussian with mean delta 
%   and variance 2*c^2, and beta has no effect.
% When alpha > 1, the mean is delta for all beta.  When alpha 
%   <= 1, the mean is undefined.
% When beta = 0, the distribution is symmetrical and delta is 
%   the median for all alpha.  When alpha = 1 and beta = 0, the 
%   distribution is Cauchy (arctangent) with median delta.
% When the submitted alpha is > 2 or < .1, or beta is outside 
%   [-1,1], an error message is generated and x is returned as a 
%   matrix of NaNs.
% Alpha < .1 is not allowed here because of the non-negligible 
%   probability of overflows.  
%
% If you're only interested in the symmetric cases, you may just 
%   set beta = 0 and skip the following considerations:
% When beta > 0 (< 0), the distribution is skewed to the right 
%   (left).
% When alpha < 1, delta, as defined above, is the unique fractile
%   that is invariant under averaging of iid contributions.  I 
%   call such a fractile a "focus of stability."  This, like the 
%   mean, is a natural location parameter.
% When alpha = 1, either every fractile is a focus of stability, 
%   as in the beta = 0 Cauchy case, or else there is no focus of 
%   stability at all, as is the case for beta ~=0.  In the latter
%   cases, which I call "afocal," delta is just an arbitrary 
%   fractile that has a simple relation to the c.f.
% When alpha > 1 and beta > 0, med(x) must lie very far below 
%   the mean as alpha approaches 1 from above.  Furthermore, as 
%   alpha approaches 1 from below, med(x) must lie very far above
%   the focus of stability when beta > 0.  If beta ~= 0, there  
%   is therefore a discontinuity in the distribution as a function
%   of alpha as alpha passes 1, when delta is held constant.
% CMS, following an insight of Vladimir Zolotarev, remove this
%   discontinuity by subtracting 
%          beta*c*tan(pi*alpha/2)
%   (equivalent to their -tan(alpha*phi0)) from x for alpha ~=1
%   in their program RSTAB, a.k.a. RNSTA in IMSL (formerly GGSTA).
%   The result is a random number whose distribution is a contin-
%   uous function of alpha, but whose location parameter (which I 
%   call zeta) is a shifted version of delta that has no known 
%   interpretation other than computational convenience.  
%   The present program restores the more meaningful "delta"  
%   parameterization by using the CMS (4.1), but with 
%   beta*c*tan(pi*alpha/2) added back in (ie with their initial
%   tan(alpha*phi0) deleted).  RNSTA therefore gives different 
%   results than the present program when beta ~= 0.  However,
%   the present beta is equivalent to the CMS beta' (BPRIME).
% Rather than using the CMS D2 and exp2 functions to compensate
%   for the ill-condition of the CMS (4.1) when alpha is very 
%   near 1, the present program merely fudges these cases by 
%   computing x from their (2.4) and adjusting for 
%   beta*c*tan(pi*alpha/2) when alpha is within 1.e-8 of 1. 
%   This should make no difference for simulation results with 
%   samples of size less than approximately 10^8, and then 
%   only when the desired alpha is within 1.e-8 of 1, but not 
%   equal to 1.
% The frequently used Gaussian and symmetric cases are coded 
%   separately so as to speed up execution.
%
% Additional references:
% V.M. Zolotarev, _One Dimensional Stable Laws_, Amer. Math. 
%   Soc., 1986.
% G. Samorodnitsky and M.S. Taqqu, _Stable Non-Gaussian Random
%   Processes_, Chapman & Hill, 1994.
% A. Janicki and A. Weron, _Simulaton and Chaotic Behavior of 
%   Alpha-Stable Stochastic Processes_, Dekker, 1994.
% J.H. McCulloch, "Financial Applications of Stable Distributons,"
%   _Handbook of Statistics_ Vol. 14, forthcoming early 1997.

% Errortraps:
if alpha < .1 | alpha > 2
    disp('Alpha must be in [.1,2] for function STABRND.')
    alpha
    x = NaN * zeros(m,n);
    return
end
if abs(beta) > 1
    disp('Beta must be in [-1,1] for function STABRND.')
    beta
    x = NaN * zeros(m,n);
    return
end

% Generate exponential w and uniform phi:
w = -log(rand(m,n));
phi = (rand(m,n)-.5)*pi;

% Gaussian case (Box-Muller):
if alpha == 2
    x = (2*sqrt(w) .* sin(phi));
    x = delta + c*x;
    return
end

% Symmetrical cases:
if beta == 0
    if alpha == 1   % Cauchy case
        x = tan(phi);
    else
        x = ((cos((1-alpha)*phi) ./ w) .^ (1/alpha - 1)    ...
            .* sin(alpha * phi) ./ cos(phi) .^ (1/alpha));
    end

    % General cases:
else
    cosphi = cos(phi);
    if abs(alpha-1) > 1.e-8
        zeta = beta * tan(pi*alpha/2);
        aphi = alpha * phi;
        a1phi = (1 - alpha) * phi;
        x = ((sin(aphi) + zeta * cos(aphi)) ./ cosphi)  ...
            .* ((cos(a1phi) + zeta * sin(a1phi))        ...
            ./ (w .* cosphi)) .^ ((1-alpha)/alpha);
    else
        bphi = (pi/2) + beta * phi;
        x = (2/pi) * (bphi .* tan(phi) - beta * log((pi/2) * w ...
            .* cosphi ./ bphi));
        if alpha ~= 1
            x = x + beta * tan(pi * alpha/2);
        end
    end
end

% Finale:
x = delta + c * x;
return
% End of STABRND.M

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % % for j=1:n,
% % %     s=nest(j,:);
    % This is a simple way of implementing Levy flights
    % For standard random walks, use step=1;
    %% Levy flights by Mantegna's algorithm
% % %     u=randn(size(s))*sigma;
% % %     v=randn(size(s));
% % %     step=u./abs(v).^(1/beta);
% % %   
    % In the next equation, the difference factor (s-best) means that 
    % when the solution is the best solution, it remains unchanged.     
% % %     stepsize=0.01*step.*(s-best);
    % Here the factor 0.01 comes from the fact that L/100 should the typical
    % step size of walks/flights where L is the typical lenghtscale; 
    % otherwise, Levy flights may become too aggresive/efficient, 
    % which makes new solutions (even) jump out side of the design domain 
    % (and thus wasting evaluations).
    % Now the actual random walks or flights
% % %     s=s+stepsize.*randn(size(s));
   % Apply simple bounds/limits
   nest(j,:)=simplebounds(s,Lb,Ub);
% % % end

%% Find the current best nest
function [fmin,best,nest,fitness]=get_best_nest(nest,newnest,fitness)
% Evaluating all new solutions
for j=1:size(nest,1),
    fnew=fobj(newnest(j,:));
    if fnew<=fitness(j),
       fitness(j)=fnew;
       nest(j,:)=newnest(j,:);
    end
end
% Find the current best
[fmin,K]=min(fitness) ;
best=nest(K,:);

%% Replace some nests by constructing new solutions/nests
function new_nest=empty_nests(nest,Lb,Ub,pa)
% A fraction of worse nests are discovered with a probability pa
n=size(nest,1);
% Discovered or not -- a status vector
K=rand(size(nest))>pa;

% In the real world, if a cuckoo's egg is very similar to a host's eggs, then 
% this cuckoo's egg is less likely to be discovered, thus the fitness should 
% be related to the difference in solutions.  Therefore, it is a good idea 
% to do a random walk in a biased way with some random step sizes.  
%% New solution by biased/selective random walks
stepsize=rand*(nest(randperm(n),:)-nest(randperm(n),:));
new_nest=nest+stepsize.*K;
for j=1:size(new_nest,1)
    s=new_nest(j,:);
  new_nest(j,:)=simplebounds(s,Lb,Ub);  
end

% Application of simple constraints
function s=simplebounds(s,Lb,Ub)
  % Apply the lower bound
  ns_tmp=s;
  I=ns_tmp<Lb;
  ns_tmp(I)=Lb(I);
  
  % Apply the upper bounds 
  J=ns_tmp>Ub;
  ns_tmp(J)=Ub(J);
  % Update this new move 
  s=ns_tmp;

%% You can replace the following by your own functions
% A d-dimensional objective function
function z=fobj(u)
%% d-dimensional sphere function sum_j=1^d (u_j-1)^2. 
%  with a minimum at (1,1, ...., 1); 
z=sum((u-1).^2);

