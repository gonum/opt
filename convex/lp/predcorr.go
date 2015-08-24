package lp

import (
	"math"

	"github.com/gonum/floats"
	"github.com/gonum/matrix/mat64"
)

func asum(x *mat64.Vector) float64 {
	return floats.Norm(x.RawVector().Data, 1)
}

func amax(x *mat64.Vector) float64 {
	return floats.Max(x.RawVector().Data)
}

func setVec(x *mat64.Vector, a float64) {
	data := x.RawVector().Data
	for i := range data {
		data[i] = a
	}
}
func addScalar(x *mat64.Vector, a float64) {
	data := x.RawVector().Data
	for i := range data {
		data[i] += a
	}
}

// Scales the rows of a linear system of equations
func scaleLinSys(A *mat64.Dense, b *mat64.Vector) (*mat64.Dense, *mat64.Vector, error) {
	m, n := A.Dims()
	for i := m - 1; i >= 0; i-- {
		a := A.RowView(i)
		alpha := mat64.Norm(a, math.Inf(1))
		if alpha == 0 {
			if b.At(i, 0) != 0 {
				return A, b, ErrInfeasible
			}
			if i < m-1 {
				A.View(i, 0, m-i, n).(*mat64.Dense).Copy(A.View(i+1, 0, m-i-1, n))
				b.ViewVec(i, m-i).CopyVec(b.ViewVec(i+1, m-i-1))
			}
			m--
			if m == 0 {
				A = mat64.NewDense(0, n, nil)
				b = mat64.NewVector(0, nil)
				return A, b, nil
			}
			A = A.View(0, 0, m, n).(*mat64.Dense)
			b = b.ViewVec(0, m)
			continue
		}
		a.ScaleVec(1/alpha, a)
		b.SetVec(i, b.At(i, 0)/alpha)
	}
	return A, b, nil
}

// Implementation of the Predictor-Corrector Interior Point algorithm to
// solve linear programs
func PredCorr(cdat []float64, Amat mat64.Matrix, bdat []float64, tol float64) (float64, []float64, []float64, []float64, error) {

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

	var mu, sigma float64

	//parameter for step size scaling
	gamma := 0.01

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
	resS := mat64.NewVector(n, nil)

	y := mat64.NewVector(m, nil)

	dx := mat64.NewVector(n, nil)
	ds := mat64.NewVector(n, nil)
	dy := mat64.NewVector(m, nil)

	dxAff := mat64.NewVector(n, nil)
	dsAff := mat64.NewVector(n, nil)
	dyAff := mat64.NewVector(m, nil)

	dxCC := mat64.NewVector(n, nil)
	dsCC := mat64.NewVector(n, nil)
	dyCC := mat64.NewVector(m, nil)

	xdivs := mat64.NewVector(n, nil)
	temp := mat64.NewDense(m, n, nil)

	lhs := mat64.NewSymDense(m, nil)
	rhs := mat64.NewVector(m, nil)

	chol := &ModChol{}

	nTemp1 := mat64.NewVector(n, nil)
	nTemp2 := mat64.NewVector(n, nil)

	alphaPrimal := 1.0
	alphaDual := 1.0
	alphaPrimalMax := 1.0
	alphaDualMax := 1.0
	ixPrimal, ixDual := 0, 0

	iter := 0
	for {
		iter++
		// resD = c - s - A**T * y
		resD.MulVec(A.T(), y)
		resD.ScaleVec(-1, resD)
		resD.AddVec(resD, c)
		resD.SubVec(resD, s)

		// resP = b - A * x
		resP.MulVec(A, x)
		resP.SubVec(b, resP)

		// resS = - x(*)s
		resS.MulElemVec(x, s)
		resS.ScaleVec(-1, resS)

		opt := mat64.Dot(c, x)
		if mat64.Norm(resP, 2)/(1+mat64.Norm(b, 2)) < tol &&
			mat64.Norm(resD, 2)/(1+mat64.Norm(c, 2)) < tol &&
			math.Abs(mat64.Dot(c, x)-mat64.Dot(b, y))/(1+mat64.Dot(c, x)) < tol {
			return opt, x.RawVector().Data, y.RawVector().Data, s.RawVector().Data, nil
		}

		if mat64.Norm(x, 1)+mat64.Norm(s, 1) > 1e6 ||
			alphaPrimal < 1e-4 || alphaDual < 1e-4 {
			return opt, x.RawVector().Data, y.RawVector().Data, s.RawVector().Data, ErrInfeasible
		}

		mu = asum(resS) / float64(n)

		//determining left hand side
		xdivs.DivElemVec(x, s)
		for i := 0; i < n; i++ {
			temp.ColView(i).ScaleVec(math.Sqrt(xdivs.At(i, 0)), A.ColView(i))
		}

		lhs.SymOuterK(1, temp)

		/*
			fmt.Println("**********************")
			fmt.Println("**   iteration", iter, "  ***")
			fmt.Println("**********************")
			fmt.Println(mat64.Formatted(x))
			fmt.Println(mat64.Formatted(s))
			fmt.Println(mat64.Formatted(y))
			fmt.Println("**********************")
			fmt.Println(mat64.Formatted(dx))
			fmt.Println(mat64.Formatted(ds))
			fmt.Println(mat64.Formatted(dy))
			fmt.Println("**********************")
			fmt.Println(mat64.Formatted(resP))
			fmt.Println(mat64.Formatted(resD))
			fmt.Println(mat64.Formatted(resS))
			fmt.Println("**********************")
			fmt.Println(chol.skipped)
			fmt.Println(alphaPrimal, alphaDual)
			fmt.Println(alphaPrimalMax, alphaDualMax)
			fmt.Println(ixPrimal, ixDual)
			fmt.Println(mat64.Dot(b, y), opt, mu)
		*/
		if len(chol.data) > 0 && math.IsNaN(chol.data[0]) {
			/*
				fmt.Println("modified cholesky failed")
				fmt.Println(mat64.Formatted(A, mat64.Squeeze()))
				fmt.Println(mat64.Formatted(b))
			*/
			return opt, x.RawVector().Data, y.RawVector().Data, s.RawVector().Data, ErrInfeasible
		}

		//factorization
		chol.From(lhs.RawSymmetric(), 1e-13)

		//right hand side
		// A*((resD+s) .* xdivs) + resP
		nTemp1.AddVec(resD, s)
		nTemp1.MulElemVec(nTemp1, xdivs)
		rhs.MulVec(A, nTemp1)
		rhs.AddVec(rhs, resP)

		//solving for dyAff
		SolveModChol(chol, rhs, dyAff)

		//calculating other steps (dxAff, dsAff)
		//dxAff = (A' * dyAff - resD - s) .* xdivs
		nTemp1.MulVec(A.T(), dyAff)
		dxAff.SubVec(nTemp1, resD)
		dxAff.SubVec(dxAff, s)
		dxAff.MulElemVec(dxAff, xdivs)

		//dsAff = -(dxAff ./ xdivs + s)
		dsAff.DivElemVec(dxAff, xdivs)
		dsAff.AddVec(dsAff, s)
		dsAff.ScaleVec(-1, dsAff)

		//determining step size
		alphaPrimal, _ = maxStep(x, dxAff)
		alphaDual, _ = maxStep(s, dsAff)

		alphaPrimal = math.Min(alphaPrimal, 1.0)
		alphaDual = math.Min(alphaDual, 1.0)

		//calculating duality gap measure for affine case
		//xt = x + alphaPrimal*dxAff
		nTemp1.ScaleVec(alphaPrimal, dxAff)
		nTemp1.AddVec(nTemp1, x)
		//st = s + alphaDual*dsAff
		nTemp2.ScaleVec(alphaDual, dsAff)
		nTemp2.AddVec(nTemp2, s)

		mu_aff := mat64.Dot(nTemp1, nTemp2) / float64(n)

		//centering parameter
		sigma = math.Pow(mu_aff/mu, 3)

		//right hand side for predictor corrector step
		// resS = sigma*mu_aff - dxAff.*dsAff
		resS.MulElemVec(dxAff, dsAff)
		resS.ScaleVec(-1, resS)
		addScalar(resS, sigma*mu_aff)

		// rhs = -A * (resS./s)
		nTemp1.DivElemVec(resS, s)
		nTemp1.ScaleVec(-1, nTemp1)
		rhs.MulVec(A, nTemp1)

		//solving for dyCC
		SolveModChol(chol, rhs, dyCC)

		//calculating other steps (dxAff, dsAff)
		nTemp1.MulVec(A.T(), dyCC)
		dxCC.MulElemVec(nTemp1, x)
		dxCC.AddVec(resS, dxCC)
		dxCC.DivElemVec(dxCC, s)

		dsCC.MulElemVec(dxCC, s)
		dsCC.SubVec(resS, dsCC)
		dsCC.DivElemVec(dsCC, x)

		dx.AddVec(dxAff, dxCC)
		dy.AddVec(dyAff, dyCC)
		ds.AddVec(dsAff, dsCC)

		//determining step size
		alphaPrimalMax, ixPrimal = maxStep(x, dx)
		alphaDualMax, ixDual = maxStep(s, ds)

		//calculating duality gap measure with full step length
		nTemp1.ScaleVec(alphaPrimalMax, dx)
		nTemp1.AddVec(nTemp1, x)
		nTemp2.ScaleVec(alphaDualMax, ds)
		nTemp2.AddVec(nTemp2, s)
		mu_f := mat64.Dot(nTemp1, nTemp2) / float64(n)
		nTemp1.MulElemVec(nTemp1, nTemp2)

		//step length calculations
		alphaPrimal = 1
		if ixPrimal >= 0 {
			fPrimal := (gamma*mu_f/(s.At(ixPrimal, 0)+alphaDualMax*ds.At(ixPrimal, 0)) - x.At(ixPrimal, 0)) / (alphaPrimalMax * dx.At(ixPrimal, 0))
			alphaPrimal = math.Max(1-gamma, fPrimal) * alphaPrimalMax
		}
		alphaDual = 1
		if ixDual >= 0 {
			fDual := (gamma*mu_f/(x.At(ixDual, 0)+alphaPrimalMax*dx.At(ixDual, 0)) - s.At(ixDual, 0)) / (alphaDualMax * ds.At(ixDual, 0))
			alphaDual = math.Max(1-gamma, fDual) * alphaDualMax
		}

		alphaPrimal = math.Min(alphaPrimal, 1)
		alphaDual = math.Min(alphaDual, 1)

		//alphaPrimal = (1 - gamma) * math.Min(alphaPrimalMax, alphaDualMax)
		//alphaDual = (1 - gamma) * math.Min(alphaPrimalMax, alphaDualMax)

		nTemp1.ScaleVec(alphaPrimal, dx)
		x.AddVec(x, nTemp1)

		nTemp1.ScaleVec(alphaDual, ds)
		s.AddVec(s, nTemp1)

		rhs.ScaleVec(alphaDual, dy)
		y.AddVec(y, rhs)
	}
}

//Maximum step-size in [0, 1] such that all elements stay positive
func maxStep(xv, dxv *mat64.Vector) (alphaMax float64, ix int) {
	x := xv.RawVector().Data
	dx := dxv.RawVector().Data
	alphaMax = 2.0
	ix = -1
	for i, d := range dx {
		if d < 0 {
			alph := -x[i] / d
			if alph < alphaMax {
				alphaMax = alph
				ix = i
			}
		}
	}
	return
}
