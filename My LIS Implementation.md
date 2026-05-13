**Process:**
1. you want $\Gamma_{pos}=(H + \Gamma_{pr}^{-1})^{-1}$ but for large $n$ this is too expensive to store and compute
2. since the data only inform a few directions of the parameter space, in all other directions the posterior is just the prior, so instead of updating all $n$ directions, we only update the $r$ directions where data actually matters
3. We need to find which directions matter: Solve the generalized eigenvalue problem $Hv =\lambda \Gamma_{pr}^{-1}v$, where the $\lambda_{i}$ measures data information vs prior information in direction $v_{i}$. Large $\lambda_{i}$ -> data dominate -> direction worth updating 
4. We cannot solve the generalized eigenvalue problem directly on large matrices - Transform it assuming square root factorization on $\Gamma_{pr}$ to reduce it to a standard eigenvalue problem, then solve that via SVD of $A_{lis}=S^{-1}_{obs}GS_{pr}$ (your singular values $\delta_{i}=\lambda_{i}$ are your eigenvalues, and $\hat{w}_{i}=S_{pr}z_{i}$ are your eigenvectors)
5. Build the approximation (Theorem 2.3) keeping only the worth $r$ eigenvectors = "optimal rank-$r$ negative update "
6. Compare $\hat{\Gamma}_{pos}$ against exact $\Gamma_{pos}$ 

**Why is it so expensive to solve inverse Problems**
Exact posterior covariance Gamma_pos = inv(H + Gamma_pr_inv)
- H = G'* Gamma_obs_inv * G costs O(n^2 * m) to form:
	- $G$ is $m \times n$ so the computation of $H$ is a matrix multiplication producing and $n \times n$ result. Each entry requires $m$ multiplications and there are $n^2$ entries $\mathcal{O}(n^2m)$
- inv(H + Gamma_pr_inv) costs  O(n^3) to compute: standard cost of an inverse with Gaussian Elimination is $\mathcal{O}(n^3)$

For large n (e.g. Jakob's FEM problem: n = d ~ thousands of DOFs):
 - O(n^3) is completely intractable 
  - G = C * K^{-1} requires solving a PDE each time G is applied: f.e. in Jakob problem $G$ is not just a random matrix, it encodes the physics, where $K$ is the FEM stiffness matrix, so $K^{-1}$ requires to solve a discretized PDE every time you apply $G$ to a vector. 
  
  LIS solution: find r << n directions where data are informative 
  - Work only in r-dimensional subspace 
  - Cost drops from O(n^3) to O(r * n^2) for covariance 
  - Cost drops from O(n^2) to O(r * n) for posterior mean 
This script demonstrates LIS on a small toy problem (n=6, m=4) to make every step inspectable. The math is identical for large n.

## 1. Matrices: 

**Forward Operator $G$**: In the bar_main the $G$ comes from FEM, it will be fixed by physics. Here with no physical problem, we generate a random matrix to stimulate "some forward operator"
rng(42) fixes the seed so every time you run the script you get the same $G$ - reproducibility 

**Gamma_obs** $\Gamma_{obs}$:
It encodes how noisy your sensors are. Diagonal means that the sensors are independent (no covariance i.e. cross relation) - must be SPD as well 
Smaller value = more precise sensor = that observation carries more weight in the likelihood (inv)
	see: $H=G^T\Gamma_{obs}^{-1}G$ meaning they push the eigenvalue $\delta_{i}$ higher, meaning the data are more informative in those directions. LIS is directly shaped by how precise your sensors are.
	Small noise variance -> large $\Gamma_{obs}^{-1}$ entries, which inflates $H$, which inflates $\delta_{i}^2$ 


## 3. Generalized Eigenvalue Problem 
**A_lis** comes from remark 4 where it is stated that: 
if $\Gamma_{pr}=S_{pr}S^T_{pr}$ and $\Gamma_{obs}=S_{obs}S^T_{obs}$ are available, then the triplets $(\delta_{i},\hat{w}_{i},\hat{v}_{i})$ come from the SVD of $S^{-1}_{obs}GS_{pr}$ which solves the eigenvalue problem $\tilde{H}w=A^TAw=\lambda w$ 

After the SVD, the $Z$ contains the right singular vectors which are the eigenvectors of the standard eigenvalue problem in the transformed space (after the change of variables $v=S_{pr}w$) but you don't want $w_{i}$, you want $\hat{w}_{i}$ the eigenvectors of the original generalized eigenvalue problem in the original parameter space.
So to undo the change of variables $\hat{w}_{i}=S_{pr}z_{i}$ 

**Singular values $\delta_{i}$**: looking at the result, the singular values are the generalized eigenvalues of the pencil $(H,\Gamma_{pr}^{-1})$, all four are greater than 1, meaning all four directions are likelihood-informed, i.e. the data update the prior in all four directions (having $\delta_{1}$ as the most informative)

## Implementation of the projector Section 3

**Questions and info of Corollary 3.1 and 3.2**
Properties of the optimal covariance approximation: 
- Describe relationship between the approximation and the directions of greatest relative reduction of prior variance
- Interpretation the covariance approximation as the result of projecting the likelihood function onto a "data-informed" subspace

**Corollary 3.1** 
We have the eigen-pair $_(\delta^2_{i},\hat{w}_{i})$ and the minimization $\mathcal{L}$ but we now have the matrix pencil $(B,\Gamma_{pos}^{-1})$ where $B \in \mathcal{M}_{r}^{-1}$ which is the class of the precision matrix $\Gamma_{pos}^{-1}$ of the form $\Gamma_{pr}^{-1}+JJ^T$ 
*So the same $\delta_{i}^2$ that minimize the covariance approximation error, also minimize the precision approximation error with the minimizer:* $$
\hat{\Gamma}_{pos}^{-1}=\Gamma_{pr}^{-1}+ \sum_{i=1}^{r}\delta_{i}^2
\tilde{w}_{i}\tilde{w}_{i}^T$$
So: $\tilde{w}_{i}$ is the natural basis for the PRECISION approximation just as $\hat{w}_{i}$ is for the covariance approximation, They are the same problem seen from two sides. 
	Why $\tilde{w}_{i}$ minimizes Rayleigh quotient: $R(z)=\frac{z^T\Gamma_{pos}z}{z^T\Gamma_{pr}z}$ is the ratio of posterior to prior variance in direction $z$, where minimizing means finding the direction where the data reduced uncertainty the most RELATIVE to the prior ~ $R(\tilde{w}_{i})=\frac{1}{1+\delta_{i}^2}$ --> small ratio, data were very informative in that direction.

**Corollary 3.2** Optimal projector
Reduced the forward operator to $\hat{G}_{r}=G\circledast P_{r}$ where $P_{r}=\sum_{i=1}^{r}\hat{w}_{i}\tilde{w}_{i}^T$ 
We get the projected model: $$
y|x \sim \mathcal{N}(GP_{r}x,\Gamma_{obs}),\space x\sim \mathcal{N}(0,\Gamma_{pr})
$$so instead of observing $Gx$ (all $n$ directions of $x$) you only observe $GP_{r}x$ meaning you first project $x$ onto the $r$-dimensional likelihood informed subspace, then apply $G$. 
	It is remarkable that the posterior covariance of this projected model is exactly $\hat{\Gamma}_{pos}$ from (2.3), instead of approximating the posterior, you approximate the forward model. 

We get to the same result:
Theorem 2.3 is the direct formula: given the eigen-pairs, build the posterior covariance approximation by substracting r rank-1 updates from the prior covariance - this is the computational recipe 
Corollary 3.2 is the interpretation: the same approximate posterior covariance is what you get if you project the forward model onto the r likelihood informed directions. Clean statistical meaning as a reduced model. 
	 When you need the posterior mean approximation of section 4, the paper shows that the optimal low rank approximation of $\mu_{pos}(y)=\Gamma_{pos}G^T\Gamma_{obs}^{-1}y$ is exactly the posterior mean of the projected model $GP_{r}$ 
## 5. Building approximate posterior covariance (2.3)
r_max = length(delta) where delta is the vector of singular values (in this case the length is 4), so the maximum rank we can go up is to 4 - we can't have more directions than singular values

```
 %% 5. build the approximate posterior covariance (Theorem 2.3)

r_max =length(delta); % maximum rank we can go up%

for r = 1:r_max %runs the approximation 4 times%

	Gamma_pos_approx = Gamma_pr; %every time we start a new rank r, you reset the approximation to Gamma_pr, we always start from the prior and substract the r terms %

		for i= 1:r %here we add one correction term at a time%

			w = W_hat(:,i); %ith eigenvector of parameter space

			d = delta(i); %ith singular value

			Gamma_pos_approx = Gamma_pos_approx - (d^2 / (1 + d^2)) * (w * w');%theorem 2.3%
			

		end

	Gamma_pos_approx_all{r} = Gamma_pos_approx; %stores the finished rank r approximation in cell array
end
```
## 6. Approximation vs real posterior covariance (Error measure)
- At first we plotted a Frobenius Norm comparing the posterior and approximate posterior, which having m=4, at r=4 one should expect a $\approx$ 0 error. The approximation formula of (2.3) is exact only when the $\hat{w}_{i}$ are perfectly orthonormal with respect to $\Gamma_{pr}^{-1}$ (in finite precision arithmetic, the SVD introduces tiny numerical errors)
	- Remark on Frobenius Norm: $\lvert \lvert A \rvert \rvert_{F}=\sqrt{\sum_{i,j}^{}a_{i,j}^2}$ which measures the total element-wise distance between two matrices.
- The paper uses the Förstner Metric (Def 2.1) which for two SPD matrices $A,B$ is: $$
d_{F}(A,B)=( \sum_{i}^{}\log^2\lambda_{i}(A,B))^{1/2}
$$ where $\lambda_{i}$ are the generalized eigenvalues of the pencil $(A,B)$. According to section 3.3 Frobenius measure absolute differences in variance, Förstner measure relative differences. If some parameters have very small prior variance, Frobenius ignores updates in those directions entirely (they are numerically small but may be exactly where the data are most informative relative to the prior). Förstner is also invariant under reparametrization

- Could be nice to have: A nice plot comparing, numerical efficiency of the Spantini optimal (2.3) against Hessian-only vs. prior only error curves
	comparing the Forstner and Frobenius 
	- Both curves decrease to 0 at r=4, at full rank the approximation is exact, as expected
	- Frobenius drops sharply, the last two directions $\delta_{3},\delta_{4}$ still carry large absolute variance, so the error only collapses when you include them (drop at r=3)
	- Forstner decreases more smoothly, because it measures relative variance reduction. Each direction contributes proportionally, so the curve is more evenly spread across all ranks 
	- **Conclusion:** @ $r=3$ you have a very small Forstner error, meaning 3 directions capture most of the information of the geometric distance. The Frobenius plot would mislead you into thinking ok $r=3$ is still bad...  **Change the Gamma obs again to 1 1 ***.   

 
