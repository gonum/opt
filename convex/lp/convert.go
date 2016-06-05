package lp

import (
	"github.com/gonum/floats"
	"github.com/gonum/matrix/mat64"
)

// Convert converts a General-form LP into a standard form LP.
// The general form of an LP is:
//  minimize c^T * x
//  s.t      G * x <= h
//           A * x = b
// And the standard form is:
//  minimize cNew^T * x
//  s.t      aNew * x = bNew
//           x >= 0
// If there are no constraints of the given type, the inputs may be nil.
func Convert(c []float64, g mat64.Matrix, h []float64, a mat64.Matrix, b []float64) (cNew []float64, aNew *mat64.Dense, bNew []float64) {
	nVar := len(c)
	nIneq := len(h)

	// Check input sizes.
	if g == nil {
		if nIneq != 0 {
			panic(badShape)
		}
	} else {
		gr, gc := g.Dims()
		if gr != nIneq {
			panic(badShape)
		}
		if gc != nVar {
			panic(badShape)
		}
	}

	nEq := len(b)
	if a == nil {
		if nEq != 0 {
			panic(badShape)
		}
	} else {
		ar, ac := a.Dims()
		if ar != nEq {
			panic(badShape)
		}
		if ac != nVar {
			panic(badShape)
		}
	}

	// Convert the general form LP.
	// Derivation:
	// 0. Start with general form
	//  min.	c^T * x
	//  s.t.	G * x <= h
	//  		A * x = b
	// 1. Introduce slack variables for each constraint
	//  min. 	c^T * x
	//  s.t.	G * x + s = h
	//			A * x = b
	//      	s >= 0
	// 2. Add non-negativity constraints for x by splitting x
	// into positive and negative components.
	//   x = xp - xn
	//   xp >= 0, xn >= 0
	// This makes the LP
	//  min.	c^T * xp - c^T xn
	//  s.t. 	G * xp - G * xn + s = h
	//			A * xp  - A * xn = b
	//			xp >= 0, xn >= 0, s >= 0
	// 3. Write the above in standard form:
	//  xt = [xp
	//	 	  xn
	//		  s ]
	//  min.	[c^T, -c^T, 0] xt
	//  s.t.	[G, -G, I] xt = h
	//   		[A, -A, 0] xt = b
	//			x >= 0

	// In summary:
	// Original LP:
	//  min.	c^T * x
	//  s.t.	G * x <= h
	//  		A * x = b
	// Standard Form:
	//  xt = [xp; xn; s]
	//  min.	[c^T, -c^T, 0] xt
	//  s.t.	[G, -G, I] xt = h
	//   		[A, -A, 0] xt = b
	//			x >= 0

	// New size of x is [xp, xn, s]
	nNewVar := nVar + nVar + nIneq

	// Construct cNew = [c; -c; 0]
	cNew = make([]float64, nNewVar)
	copy(cNew, c)
	copy(cNew[nVar:], c)
	floats.Scale(-1, cNew[nVar:2*nVar])

	// New number of equality constraints is the number of total constraints.
	nNewEq := nIneq + nEq

	// Construct bNew = [h, b].
	bNew = make([]float64, nNewEq)
	copy(bNew, h)
	copy(bNew[nIneq:], b)

	// Construct aNew = [G, -G, I; A, -A, 0].
	aNew = mat64.NewDense(nNewEq, nNewVar, nil)
	if nIneq != 0 {
		aView := (aNew.View(0, 0, nIneq, nVar)).(*mat64.Dense)
		aView.Copy(g)
		aView = (aNew.View(0, nVar, nIneq, nVar)).(*mat64.Dense)
		aView.Scale(-1, g)
		aView = (aNew.View(0, 2*nVar, nIneq, nIneq)).(*mat64.Dense)
		for i := 0; i < nIneq; i++ {
			aView.Set(i, i, 1)
		}
	}
	if nEq != 0 {
		aView := (aNew.View(nIneq, 0, nEq, nVar)).(*mat64.Dense)
		aView.Copy(a)
		aView = (aNew.View(nIneq, nVar, nEq, nVar)).(*mat64.Dense)
		aView.Scale(-1, a)
	}
	return cNew, aNew, bNew
}
