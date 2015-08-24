package lp

import (
	"fmt"
	"math"

	"github.com/gonum/matrix/mat64"
)

// Solves a linear program by solving a self-dual linear program with the
// same solution.
// Returns optimal value as well as optimal primal and dual variables.
func SelfDual(cdat []float64, Amat mat64.Matrix, bdat []float64, tol float64) (float64, []float64, []float64, []float64, error) {

	m, n := Amat.Dims()
	A := mat64.NewDense(m, n, nil)
	A.Clone(Amat)

	err := verifyInputs(nil, cdat, A, bdat)
	if err != nil {
		if (err == ErrUnbounded) || (err == ErrZeroRow) || (err == ErrZeroColumn) {
			//return math.Inf(-1), nil, nil, nil, ErrUnbounded
		} else {
			return math.NaN(), nil, nil, nil, err
		}
	}

	// parameter for affine step weighting
	const beta1 = 0.1
	// parameters for step size scaling
	const beta2 = 1e-8
	const beta3 = 0.9999
	// residual tolerances
	rhoP := 1e-8
	rhoD := 1e-8
	rhoG := 1e-8
	const rhoMu = 1e-10
	const rhoI = 1e-10

	c := mat64.NewVector(n, cdat)

	btmp := make([]float64, m)
	copy(btmp, bdat)
	b := mat64.NewVector(m, btmp)

	A, b, err = scaleLinSys(A, b)
	if err != nil {
		return math.NaN(), nil, nil, nil, err
	}
	m, n = A.Dims()

	x := mat64.NewVector(n, nil)
	setVec(x, 1)

	s := mat64.NewVector(n, nil)
	setVec(s, 1)

	if m == 0 {
		for _, v := range cdat {
			if v < 0 {
				return math.Inf(-1), nil, nil, nil, ErrUnbounded
			}
		}
		s.CopyVec(c)
		return 0, x.RawVector().Data, nil, s.RawVector().Data, nil
	}

	resD := mat64.NewVector(n, nil)
	resP := mat64.NewVector(m, nil)
	resXS := mat64.NewVector(n, nil)

	y := mat64.NewVector(m, nil)

	dx := mat64.NewVector(n, nil)
	ds := mat64.NewVector(n, nil)
	dy := mat64.NewVector(m, nil)

	p := mat64.NewVector(n, nil)
	q := mat64.NewVector(m, nil)

	xdivs := mat64.NewVector(n, nil)
	temp := mat64.NewDense(m, n, nil)

	lhs := mat64.NewSymDense(m, nil)
	rhs := mat64.NewVector(m, nil)

	chol := &ModChol{}

	nTemp1 := mat64.NewVector(n, nil)
	nTemp2 := mat64.NewVector(n, nil)

	k, t := 1.0, 1.0
	dk, dt := 0.0, 0.0

	alpha := 1.0

	iter := 0
	for {
		iter++
		//fmt.Println("iteration", iter)

		// residuals for full HSD system

		primal := mat64.Dot(c, x)
		dual := mat64.Dot(b, y)
		resG := k - dual + primal

		// resP = t*b - A*x
		resP.MulVec(A, x)
		resP.ScaleVec(-1, resP)
		resP.AddScaledVec(resP, t, b)

		// resD = c*t - A^T*y - s
		resD.MulVec(A.T(), y)
		resD.ScaleVec(-1, resD)
		resD.SubVec(resD, s)
		resD.AddScaledVec(resD, t, c)

		resXS.MulElemVec(x, s)
		resXS.ScaleVec(-1, resXS)

		resTK := -t * k

		mu := (asum(resXS) + math.Abs(t*k)) / float64(n+1)

		/*
			fmt.Println("*** variables ***")
			fmt.Println(mat64.Formatted(x))
			fmt.Println(mat64.Formatted(y))
			fmt.Println(mat64.Formatted(s))
			fmt.Println(t, k)
			fmt.Println("*** residuals ***")
			fmt.Println(mat64.Formatted(resP))
			fmt.Println(mat64.Formatted(resD))
			fmt.Println(resG)
			fmt.Println(mat64.Formatted(resXS))
			fmt.Println(resTK)
			fmt.Println("*** opt. cond. ***")
			fmt.Println(mat64.Norm(resP, 1), rhoP)
			fmt.Println(mat64.Norm(resD, 1), rhoD)
			fmt.Println(math.Abs(resG), rhoG)
			fmt.Println(t, rhoI, rhoI*k)
			fmt.Println(primal, dual, primal-dual)
		*/

		if iter == 1 {
			// scale tolerances with initial residuals
			rhoP = rhoP * math.Max(1, mat64.Norm(resP, 1))
			rhoD = rhoD * math.Max(1, mat64.Norm(resD, 1))
			rhoG = rhoG * math.Max(1, math.Abs(resG))
		}

		// If a row which is skipped by modified cholesky leads to
		// a non-zero residual, Ax=b has no solution and the
		// problem is infeasible
		sumResP := mat64.Norm(resP, 1)
		resPskipped := sumResP
		for _, i := range chol.skipped {
			resPskipped -= math.Abs(resP.At(i, 0))
		}

		if resPskipped < rhoP && mat64.Norm(resD, 1) < rhoD {
			if math.Abs(primal-dual)/(t+math.Abs(dual)) < tol {
				if sumResP < rhoP {
					// feasible solution
					x.ScaleVec(1./t, x)
					y.ScaleVec(1./t, y)
					s.ScaleVec(1./t, s)
					return primal / t, x.RawVector().Data, y.RawVector().Data, s.RawVector().Data, nil
				}
				return primal, x.RawVector().Data, y.RawVector().Data, s.RawVector().Data, ErrInfeasible
			}

			if math.Abs(resG) < rhoG && t < rhoI*math.Max(1, k) {
				// infeasible
				if primal < 0 {
					return primal, x.RawVector().Data, y.RawVector().Data, s.RawVector().Data, ErrInfeasible
				} else {
					return primal, x.RawVector().Data, y.RawVector().Data, s.RawVector().Data, ErrInfeasible
				}
			}
		}

		if t < rhoI*math.Min(1, k) &&
			mu < rhoMu {
			fmt.Println("ill-posed", mu, tol, t, k)
			return primal, x.RawVector().Data, y.RawVector().Data, s.RawVector().Data, ErrInfeasible
		}

		//determining left hand side
		xdivs.DivElemVec(x, s)
		for i := 0; i < n; i++ {
			temp.ColView(i).ScaleVec(math.Sqrt(xdivs.At(i, 0)), A.ColView(i))
		}

		lhs.SymOuterK(1, temp)

		if iter > 100 {
			fmt.Println("Too many iterations ... ")
			fmt.Println(mat64.Formatted(A, mat64.Squeeze()))
			fmt.Println(mat64.Formatted(b))
			return primal, x.RawVector().Data, y.RawVector().Data, s.RawVector().Data, ErrInfeasible
		}
		if len(chol.data) > 0 && math.IsNaN(chol.data[0]) {
			fmt.Println("modified cholesky failed (this should not happen)")
			fmt.Println(mat64.Formatted(A, mat64.Squeeze()))
			fmt.Println(mat64.Formatted(b, mat64.Squeeze()))
			fmt.Println("alpha", alpha)

			panic("etc")
			return primal, x.RawVector().Data, y.RawVector().Data, s.RawVector().Data, ErrInfeasible
		}

		//factorization
		chol.From(lhs.RawSymmetric(), 1e-13)

		//fmt.Println("chol skipped", chol.skipped)

		//right hand side for correction with [p; q]
		// rhs = b + A*(c .* x./s)
		nTemp1.MulElemVec(c, xdivs)
		rhs.MulVec(A, nTemp1)
		rhs.AddVec(rhs, b)

		//solving for q
		SolveModChol(chol, rhs, q)

		//calculating p = (A^T*q - c) .* x./s
		nTemp1.MulVec(A.T(), q)
		p.SubVec(nTemp1, c)
		p.MulElemVec(p, xdivs)

		eta := 1.0
		gamma := 0.0

		for step := 0; step < 2; step++ {
			if step == 1 {
				// Adjust residuals for predictor corrector step
				resP.ScaleVec(eta, resP)
				resD.ScaleVec(eta, resD)
				resG = eta * resG

				resTK = -t*k + gamma*mu - dt*dk
				// resXS = -x.*s + gamma*mu*1 - dx.*ds
				nTemp1.MulElemVec(dx, ds)
				addScalar(resXS, gamma*mu)
				resXS.SubVec(resXS, nTemp1)
			} else {
			}

			// right hand side for affine step
			// A*((resD - rxs./x) .* xdivs) + resP
			nTemp2.DivElemVec(resXS, x)
			nTemp1.SubVec(resD, nTemp2)
			nTemp1.MulElemVec(nTemp1, xdivs)
			rhs.MulVec(A, nTemp1)
			rhs.AddVec(rhs, resP)

			// solve for dy
			SolveModChol(chol, rhs, dy)
			// calculate dx = (A^T*dy - resD + resXS./x) .* x./s
			nTemp1.MulVec(A.T(), dy)
			dx.SubVec(nTemp1, resD)
			dx.AddVec(dx, nTemp2) // nTemp2 = rxs./x see above
			dx.MulElemVec(dx, xdivs)

			dt = resG + resTK/t + mat64.Dot(c, dx) - mat64.Dot(b, dy)
			dt = dt / (k/t - mat64.Dot(c, p) + mat64.Dot(b, q))

			// correction
			dx.AddScaledVec(dx, dt, p)
			dy.AddScaledVec(dy, dt, q)

			// calculate ds = (rxs - dx.*s) ./ x
			ds.MulElemVec(dx, s)
			ds.SubVec(resXS, ds)
			ds.DivElemVec(ds, x)

			dk = (resTK - k*dt) / t

			//determining step size
			alphaX, _ := maxStep(x, dx)
			alphaS, _ := maxStep(s, ds)
			alphaT := math.Max(1.0, -t/dt)
			alphaK := math.Max(1.0, -k/dk)

			//fmt.Println("alphas:", alphaX, alphaS, alphaT, alphaK)

			alphaP := beta3 * math.Min(math.Min(alphaX, alphaT), alphaK)
			alphaD := beta3 * math.Min(math.Min(alphaS, alphaT), alphaK)

			alpha = math.Min(alphaP, alphaD)
			alpha = math.Min(alpha, 1.0)

			gamma = (1 - alpha) * (1 - alpha) * math.Min(1-alpha, beta1)
			eta = 1 - gamma

		}

		//fmt.Println("alpha pre", alpha)
		for j := 0; j < 10; j++ {
			nTemp1.AddScaledVec(x, alpha, dx)
			nTemp2.AddScaledVec(s, alpha, ds)

			tt := t + alpha*dt
			kt := k + alpha*dk

			mu := (mat64.Dot(nTemp1, nTemp2) + tt*kt) / float64(n+1)

			if tt*kt < beta2*mu {
				alpha = 0.9 * alpha
				continue
			}
			for i := 0; i < n; i++ {
				if nTemp1.At(i, 0)*nTemp2.At(i, 0) < beta2*mu {
					alpha = 0.9 * alpha
					continue
				}
			}
			break
		}
		//fmt.Println("alpha post", alpha)

		/*
			fmt.Println("*** steps ***")
			fmt.Println(mat64.Formatted(dx))
			fmt.Println(mat64.Formatted(dy))
			fmt.Println(mat64.Formatted(ds))
			fmt.Println(dt, dk)
		*/

		x.AddScaledVec(x, alpha, dx)
		s.AddScaledVec(s, alpha, ds)
		y.AddScaledVec(y, alpha, dy)
		t = t + alpha*dt
		k = k + alpha*dk
	}
}
