package lp

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/gonum/matrix/mat64"
)

func randSlice(n int) []float64 {
	x := make([]float64, n)
	for i := range x {
		v := rand.Float64()
		if v > 0.3 {
			x[i] = rand.NormFloat64()
		} else {
			x[i] = 0
		}
	}
	return x
}

func TestSmall(t *testing.T) {
	m := 2
	n := 3
	tol := 1e-6

	A := mat64.NewDense(m, n, []float64{1, 2, 3, 4, 5, 6})
	cdat := []float64{3, 4, 5}
	xdat := []float64{1, 1, 1}

	xt := mat64.NewVector(n, xdat)
	bt := mat64.NewVector(m, nil)
	bt.MulVec(A, xt)
	bdat := bt.RawVector().Data
	//bdat = randSlice(m)

	primalpc, xdatpc, ydat, sdat, err := PredCorr(cdat, A, bdat, tol)
	primal, xdat, ydat, sdat, err := SelfDual(cdat, A, bdat, tol)
	fmt.Println(primalpc, xdatpc)
	fmt.Println(primal, xdat)

	var dev float64
	if err == nil {
		m, n = A.Dims()

		resD := mat64.NewVector(n, nil)
		resP := mat64.NewVector(m, nil)
		resS := mat64.NewVector(n, nil)

		x := mat64.NewVector(n, xdat)
		y := mat64.NewVector(m, ydat)
		s := mat64.NewVector(n, sdat)

		c := mat64.NewVector(n, cdat)
		b := mat64.NewVector(m, bdat[:m])

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

		dev = (asum(resD) + asum(resP) + asum(resS)) / float64(n)
	}
	if (dev >= tol) && err == nil {
		t.Error(dev)
		t.Fail()
	}
}

func TestLinprog(t *testing.T) {

	noTests := 100

	infeasible := 0
	unbounded := 0
	bounded := 0
	singular := 0
	zeroRowCol := 0
	bad := 0

	for test := 0; test < noTests; test++ {
		m := 4
		n := 6
		tol := 1e-8

		A := mat64.NewDense(m, n, randSlice(m*n))
		cdat := randSlice(n)

		xdat := randSlice(n)
		for i := range xdat {
			if xdat[i] < 0 {
				xdat[i] = 0
			}
		}
		xt := mat64.NewVector(n, xdat)
		bt := mat64.NewVector(m, nil)
		bt.MulVec(A, xt)

		A, bt, err := scaleLinSys(A, bt)
		if err != nil {
			infeasible++
			continue
		}
		bdat := bt.RawVector().Data[:bt.Len()]

		//bdat = randSlice(m)

		_, xdat, ydat, sdat, err := SelfDual(cdat, A, bdat, tol)

		primalInfeasible := err == ErrInfeasible
		primalUnbounded := err == ErrUnbounded
		primalBounded := err == nil
		primalASingular := err == ErrSingular
		primalZero := err == ErrZeroRow || err == ErrZeroColumn

		switch {
		case primalInfeasible:
			infeasible++
		case primalUnbounded:
			unbounded++
		case primalBounded:
			bounded++
		case primalASingular:
			singular++
		case primalZero:
			zeroRowCol++
		default:
			bad++
		}

		var dev float64
		if err == nil {
			m, n = A.Dims()

			resD := mat64.NewVector(n, nil)
			resP := mat64.NewVector(m, nil)
			resS := mat64.NewVector(n, nil)

			x := mat64.NewVector(n, xdat)
			y := mat64.NewVector(m, ydat)
			s := mat64.NewVector(n, sdat)

			c := mat64.NewVector(n, cdat)
			b := mat64.NewVector(m, bdat[:m])

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

			dev = (asum(resD) + asum(resP) + asum(resS)) / float64(n)
		}
		if (dev >= 500*tol) && err == nil {
			t.Error(dev)
			t.Fail()
		}
	}

	fmt.Println("Cases: ", noTests)
	fmt.Println("Bounded:", bounded)
	fmt.Println("Singular:", singular)
	fmt.Println("Unbounded:", unbounded)
	fmt.Println("Infeasible:", infeasible)
	fmt.Println("Zero Row/col:", zeroRowCol)
	fmt.Println("Bad:", bad)
}
