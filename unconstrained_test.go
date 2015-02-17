// Copyright ©2014 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package optimize

import (
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"testing"

	"github.com/gonum/floats"
	funcs "github.com/gonum/optimize/functions"
)

type unconstrainedTest struct {
	// f is the function that is being minimized.
	f Function
	// x is the initial guess.
	x []float64
	// gradTol is the absolute gradient tolerance for the test. If gradTol == 0,
	// the default tolerance 1e-12 will be used.
	gradTol float64
	// long indicates that the test takes long time to finish and will be
	// excluded if testing.Short() is true.
	long bool
}

func (t unconstrainedTest) String() string {
	dim := len(t.x)
	if dim <= 10 {
		// Print the initial X only for small-dimensional problems.
		return fmt.Sprintf("F: %v\nDim: %v\nInitial X: %v\nGradientAbsTol: %v",
			reflect.TypeOf(t.f), dim, t.x, t.gradTol)
	}
	return fmt.Sprintf("F: %v\nDim: %v\nGradientAbsTol: %v",
		reflect.TypeOf(t.f), dim, t.gradTol)
}

var gradientDescentTests = []unconstrainedTest{
	{
		f: funcs.Beale{},
		x: []float64{1, 1},
	},
	{
		f: funcs.Beale{},
		x: []float64{3.00001, 0.50001},
	},
	{
		f: funcs.BiggsEXP2{},
		x: []float64{1, 2},
	},
	{
		f: funcs.BiggsEXP2{},
		x: []float64{1.00001, 10.00001},
	},
	{
		f: funcs.BiggsEXP3{},
		x: []float64{1, 2, 1},
	},
	{
		f: funcs.BiggsEXP3{},
		x: []float64{1.00001, 10.00001, 3.00001},
	},
	{
		f: funcs.ExtendedRosenbrock{},
		x: []float64{-1.2, 1},
	},
	{
		f: funcs.ExtendedRosenbrock{},
		x: []float64{1.00001, 1.00001},
	},
	{
		f: funcs.ExtendedRosenbrock{},
		x: []float64{-1.2, 1, -1.2},
	},
	{
		f:    funcs.ExtendedRosenbrock{},
		x:    []float64{-120, 100, 50},
		long: true,
	},
	{
		f: funcs.ExtendedRosenbrock{},
		x: []float64{1, 1, 1},
	},
	{
		f:       funcs.ExtendedRosenbrock{},
		x:       []float64{1.00001, 1.00001, 1.00001},
		gradTol: 1e-8,
	},
	{
		f:       funcs.Gaussian{},
		x:       []float64{0.4, 1, 0},
		gradTol: 1e-9,
	},
	{
		f:       funcs.Gaussian{},
		x:       []float64{0.3989561, 1.0000191, 0},
		gradTol: 1e-9,
	},
	{
		f: funcs.HelicalValley{},
		x: []float64{-1, 0, 0},
	},
	{
		f: funcs.HelicalValley{},
		x: []float64{1.00001, 0.00001, 0.00001},
	},
	{
		f:       funcs.Trigonometric{},
		x:       []float64{0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1},
		gradTol: 1e-8,
	},
	{
		f: funcs.Trigonometric{},
		x: []float64{0.042964, 0.043976, 0.045093, 0.046338, 0.047744,
			0.049354, 0.051237, 0.195209, 0.164977, 0.060148},
		gradTol: 1e-8,
	},
	newVariablyDimensioned(2, 0),
	{
		f: funcs.VariablyDimensioned{},
		x: []float64{1.00001, 1.00001},
	},
	newVariablyDimensioned(10, 0),
	{
		f: funcs.VariablyDimensioned{},
		x: []float64{1.00001, 1.00001, 1.00001, 1.00001, 1.00001, 1.00001, 1.00001, 1.00001, 1.00001, 1.00001},
	},
}

var cgTests = []unconstrainedTest{
	{
		f: funcs.BiggsEXP4{},
		x: []float64{1, 2, 1, 1},
	},
	{
		f: funcs.BiggsEXP4{},
		x: []float64{1.00001, 10.00001, 1.00001, 5.00001},
	},
	{
		f:       funcs.BiggsEXP5{},
		x:       []float64{1, 2, 1, 1, 1},
		gradTol: 1e-7,
	},
	{
		f: funcs.BiggsEXP5{},
		x: []float64{1.00001, 10.00001, 1.00001, 5.00001, 4.00001},
	},
	{
		f:       funcs.BiggsEXP6{},
		x:       []float64{1, 2, 1, 1, 1, 1},
		gradTol: 1e-7,
	},
	{
		f:       funcs.BiggsEXP6{},
		x:       []float64{1.00001, 10.00001, 1.00001, 5.00001, 4.00001, 3.00001},
		gradTol: 1e-8,
	},
	// TODO(vladimir-ch): Enable this test when angle restart condition in CG
	// has been revised.
	// {
	// 	f:       funcs.Box3D{},
	// 	x:       []float64{0, 10, 20},
	// },
	{
		f: funcs.Box3D{},
		x: []float64{1.00001, 10.00001, 1.00001},
	},
	{
		f: funcs.Box3D{},
		x: []float64{100.00001, 100.00001, 0.00001},
	},
	{
		f: funcs.ExtendedPowellSingular{},
		x: []float64{3, -1, 0, 3},
	},
	{
		f: funcs.ExtendedPowellSingular{},
		x: []float64{0.00001, 0.00001, 0.00001, 0.00001},
	},
	{
		f:       funcs.ExtendedPowellSingular{},
		x:       []float64{3, -1, 0, 3, 3, -1, 0, 3},
		gradTol: 1e-8,
	},
	{
		f: funcs.ExtendedPowellSingular{},
		x: []float64{0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001},
	},
	{
		f: funcs.ExtendedRosenbrock{},
		x: []float64{-1.2, 1, -1.2, 1},
	},
	{
		f: funcs.ExtendedRosenbrock{},
		x: []float64{1e4, 1e4},
	},
	{
		f: funcs.ExtendedRosenbrock{},
		x: []float64{1.00001, 1.00001, 1.00001, 1.00001},
	},
	{
		f:       funcs.PenaltyI{},
		x:       []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
		gradTol: 1e-10,
	},
	{
		f:       funcs.PenaltyI{},
		x:       []float64{0.250007, 0.250007, 0.250007, 0.250007},
		gradTol: 1e-10,
	},
	{
		f: funcs.PenaltyI{},
		x: []float64{0.1581, 0.1581, 0.1581, 0.1581, 0.1581, 0.1581,
			0.1581, 0.1581, 0.1581, 0.1581},
		gradTol: 1e-10,
	},
	{
		f:       funcs.PenaltyII{},
		x:       []float64{0.5, 0.5, 0.5, 0.5},
		gradTol: 1e-8,
	},
	{
		f:       funcs.PenaltyII{},
		x:       []float64{0.19999, 0.19131, 0.4801, 0.51884},
		gradTol: 1e-8,
	},
	{
		f: funcs.PenaltyII{},
		x: []float64{0.19998, 0.01035, 0.01960, 0.03208, 0.04993, 0.07651,
			0.11862, 0.19214, 0.34732, 0.36916},
		gradTol: 1e-6,
	},
	{
		f:       funcs.PowellBadlyScaled{},
		x:       []float64{1.09815e-05, 9.10614},
		gradTol: 1e-8,
	},
	newVariablyDimensioned(100, 1e-10),
	newVariablyDimensioned(1000, 1e-9),
	newVariablyDimensioned(10000, 1e-7),
	{
		f:       funcs.Watson{},
		x:       []float64{0, 0, 0, 0, 0, 0},
		gradTol: 1e-7,
	},
	{
		f:       funcs.Watson{},
		x:       []float64{-0.01572, 1.01243, -0.23299, 1.26043, -1.51372, 0.99299},
		gradTol: 1e-7,
	},
	{
		f:       funcs.Watson{},
		x:       []float64{0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
		gradTol: 1e-7,
		long:    true,
	},
	{
		f: funcs.Watson{},
		x: []float64{-1.53070e-05, 0.99978, 0.01476, 0.14634, 1.00082,
			-2.61773, 4.10440, -3.14361, 1.05262},
		gradTol: 1e-7,
	},
	{
		f:       funcs.Wood{},
		x:       []float64{-3, -1, -3, -1},
		gradTol: 1e-6,
	},
}

var newtonTests = []unconstrainedTest{
	{
		f: funcs.BiggsEXP4{},
		x: []float64{1, 2, 1, 1},
	},
	{
		f: funcs.BiggsEXP4{},
		x: []float64{1.00001, 10.00001, 1.00001, 5.00001},
	},
	{
		f: funcs.BiggsEXP5{},
		x: []float64{1, 2, 1, 1, 1},
	},
	{
		f: funcs.BiggsEXP5{},
		x: []float64{1.00001, 10.00001, 1.00001, 5.00001, 4.00001},
	},
	{
		f:       funcs.BiggsEXP6{},
		x:       []float64{1, 2, 1, 1, 1, 1},
		gradTol: 1e-8,
	},
	{
		f:       funcs.BiggsEXP6{},
		x:       []float64{1.00001, 10.00001, 1.00001, 5.00001, 4.00001, 3.00001},
		gradTol: 1e-8,
	},
	{
		f: funcs.Box3D{},
		x: []float64{0, 10, 20},
	},
	{
		f: funcs.Box3D{},
		x: []float64{1.00001, 10.00001, 1.00001},
	},
	{
		f: funcs.Box3D{},
		x: []float64{100.00001, 100.00001, 0.00001},
	},
	{
		f: funcs.BrownBadlyScaled{},
		x: []float64{1, 1},
	},
	{
		f: funcs.BrownBadlyScaled{},
		x: []float64{1.000001e6, 2.01e-6},
	},
	{
		f: funcs.ExtendedPowellSingular{},
		x: []float64{3, -1, 0, 3},
	},
	{
		f: funcs.ExtendedPowellSingular{},
		x: []float64{0.00001, 0.00001, 0.00001, 0.00001},
	},
	{
		f: funcs.ExtendedPowellSingular{},
		x: []float64{3, -1, 0, 3, 3, -1, 0, 3},
	},
	{
		f: funcs.ExtendedPowellSingular{},
		x: []float64{0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001},
	},
	{
		f: funcs.ExtendedRosenbrock{},
		x: []float64{-1.2, 1, -1.2, 1},
	},
	{
		f: funcs.ExtendedRosenbrock{},
		x: []float64{1.00001, 1.00001, 1.00001, 1.00001},
	},
	{
		f:       funcs.Gaussian{},
		x:       []float64{0.4, 1, 0},
		gradTol: 1e-11,
	},
	{
		f: funcs.GulfResearchAndDevelopment{},
		x: []float64{5, 2.5, 0.15},
	},
	{
		f: funcs.GulfResearchAndDevelopment{},
		x: []float64{50.00001, 25.00001, 1.50001},
	},
	{
		f: funcs.GulfResearchAndDevelopment{},
		x: []float64{99.89529, 60.61453, 9.16124},
	},
	{
		f: funcs.GulfResearchAndDevelopment{},
		x: []float64{201.66258, 60.61633, 10.22489},
	},
	{
		f: funcs.PenaltyI{},
		x: []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
	},
	{
		f: funcs.PenaltyI{},
		x: []float64{0.250007, 0.250007, 0.250007, 0.250007},
	},
	{
		f: funcs.PenaltyI{},
		x: []float64{0.1581, 0.1581, 0.1581, 0.1581, 0.1581, 0.1581,
			0.1581, 0.1581, 0.1581, 0.1581},
	},
	{
		f:       funcs.PenaltyII{},
		x:       []float64{0.5, 0.5, 0.5, 0.5},
		gradTol: 1e-10,
	},
	{
		f:       funcs.PenaltyII{},
		x:       []float64{0.19999, 0.19131, 0.4801, 0.51884},
		gradTol: 1e-10,
	},
	{
		f:       funcs.PenaltyII{},
		x:       []float64{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5},
		gradTol: 1e-9,
	},
	{
		f: funcs.PenaltyII{},
		x: []float64{0.19998, 0.01035, 0.01960, 0.03208, 0.04993, 0.07651,
			0.11862, 0.19214, 0.34732, 0.36916},
		gradTol: 1e-9,
	},
	{
		f: funcs.PowellBadlyScaled{},
		x: []float64{0, 1},
	},
	{
		f:       funcs.PowellBadlyScaled{},
		x:       []float64{1.09815e-05, 9.10614},
		gradTol: 1e-10,
	},
	newVariablyDimensioned(100, 0),
	{
		f:       funcs.Watson{},
		x:       []float64{0, 0, 0, 0, 0, 0},
		gradTol: 1e-7,
	},
	{
		f:       funcs.Watson{},
		x:       []float64{-0.01572, 1.01243, -0.23299, 1.26043, -1.51372, 0.99299},
		gradTol: 1e-7,
	},
	{
		f:       funcs.Watson{},
		x:       []float64{0, 0, 0, 0, 0, 0, 0, 0, 0},
		gradTol: 1e-8,
	},
	{
		f: funcs.Watson{},
		x: []float64{-1.53070e-05, 0.99978, 0.01476, 0.14634, 1.00082,
			-2.61773, 4.10440, -3.14361, 1.05262},
		gradTol: 1e-8,
	},
}

var bfgsTests = []unconstrainedTest{
	{
		f:       funcs.BiggsEXP6{},
		x:       []float64{1, 2, 1, 1, 1, 1},
		gradTol: 1e-10,
	},
	{
		f:       funcs.BiggsEXP6{},
		x:       []float64{1.00001, 10.00001, 1.00001, 5.00001, 4.00001, 3.00001},
		gradTol: 1e-10,
	},
	{
		f:       funcs.BrownAndDennis{},
		x:       []float64{25, 5, -5, -1},
		gradTol: 1e-5,
	},
	{
		f: funcs.ExtendedRosenbrock{},
		x: []float64{1e5, 1e5},
	},
	{
		f:       funcs.Gaussian{},
		x:       []float64{0.398, 1, 0},
		gradTol: 1e-11,
	},
	{
		f: funcs.Wood{},
		x: []float64{-3, -1, -3, -1},
	},
}

var lbfgsTests = []unconstrainedTest{
	{
		f:       funcs.BiggsEXP6{},
		x:       []float64{1, 2, 1, 1, 1, 1},
		gradTol: 1e-8,
	},
	{
		f:       funcs.BiggsEXP6{},
		x:       []float64{1.00001, 10.00001, 1.00001, 5.00001, 4.00001, 3.00001},
		gradTol: 1e-8,
	},
	{
		f: funcs.ExtendedRosenbrock{},
		x: []float64{1e7, 1e7},
	},
	{
		f:       funcs.Gaussian{},
		x:       []float64{0.398, 1, 0},
		gradTol: 1e-10,
	},
	newVariablyDimensioned(1000, 1e-10),
	newVariablyDimensioned(10000, 1e-8),
}

func newVariablyDimensioned(dim int, gradTol float64) unconstrainedTest {
	x := make([]float64, dim)
	for i := range x {
		x[i] = float64(dim-i-1) / float64(dim)
	}
	return unconstrainedTest{
		f:       funcs.VariablyDimensioned{},
		x:       x,
		gradTol: gradTol,
	}
}

func TestLocal(t *testing.T) {
	// TODO: When method is nil, Local chooses the method automatically. At
	// present, it always chooses BFGS (or panics if the function does not
	// implement Df() or FDf()). For now, run this test with the simplest set
	// of problems and revisit this later when more methods are added.
	testLocal(t, gradientDescentTests, nil)
}

func TestGradientDescent(t *testing.T) {
	method := &GradientDescent{}
	testLocal(t, gradientDescentTests, method)
	testLocalFromMinima(t, method)
}

func TestGradientDescentBacktracking(t *testing.T) {
	method := &GradientDescent{
		LinesearchMethod: &Backtracking{
			FunConst: 0.1,
		},
	}
	testLocal(t, gradientDescentTests, method)
	testLocalFromMinima(t, method)
}

func TestGradientDescentBisection(t *testing.T) {
	method := &GradientDescent{
		LinesearchMethod: &Bisection{},
	}
	testLocal(t, gradientDescentTests, method)
	testLocalFromMinima(t, method)
}

func TestCG(t *testing.T) {
	var tests []unconstrainedTest
	tests = append(tests, gradientDescentTests...)
	tests = append(tests, cgTests...)
	method := &CG{}
	testLocal(t, tests, method)
	testLocalFromMinima(t, method)
}

func TestFletcherReevesQuadStep(t *testing.T) {
	var tests []unconstrainedTest
	tests = append(tests, gradientDescentTests...)
	tests = append(tests, cgTests...)
	method := &CG{
		Variant:     &FletcherReeves{},
		InitialStep: &QuadraticStepSize{},
	}
	testLocal(t, tests, method)
	testLocalFromMinima(t, method)
}

func TestFletcherReevesFirstOrderStep(t *testing.T) {
	var tests []unconstrainedTest
	tests = append(tests, gradientDescentTests...)
	tests = append(tests, cgTests...)
	method := &CG{
		Variant:     &FletcherReeves{},
		InitialStep: &FirstOrderStepSize{},
	}
	testLocal(t, tests, method)
	testLocalFromMinima(t, method)
}

func TestHestenesStiefelQuadStep(t *testing.T) {
	var tests []unconstrainedTest
	tests = append(tests, gradientDescentTests...)
	tests = append(tests, cgTests...)
	method := &CG{
		Variant:     &HestenesStiefel{},
		InitialStep: &QuadraticStepSize{},
	}
	testLocal(t, tests, method)
	testLocalFromMinima(t, method)
}

func TestHestenesStiefelFirstOrderStep(t *testing.T) {
	var tests []unconstrainedTest
	tests = append(tests, gradientDescentTests...)
	tests = append(tests, cgTests...)
	method := &CG{
		Variant:     &HestenesStiefel{},
		InitialStep: &FirstOrderStepSize{},
	}
	testLocal(t, tests, method)
	testLocalFromMinima(t, method)
}

func TestPolakRibiereQuadStep(t *testing.T) {
	var tests []unconstrainedTest
	tests = append(tests, gradientDescentTests...)
	tests = append(tests, cgTests...)
	method := &CG{
		Variant:     &PolakRibierePolyak{},
		InitialStep: &QuadraticStepSize{},
	}
	testLocal(t, tests, method)
	testLocalFromMinima(t, method)
}

func TestPolakRibiereFirstOrderStep(t *testing.T) {
	var tests []unconstrainedTest
	tests = append(tests, gradientDescentTests...)
	tests = append(tests, cgTests...)
	method := &CG{
		Variant:     &PolakRibierePolyak{},
		InitialStep: &FirstOrderStepSize{},
	}
	testLocal(t, tests, method)
	testLocalFromMinima(t, method)
}

func TestDaiYuanQuadStep(t *testing.T) {
	var tests []unconstrainedTest
	tests = append(tests, gradientDescentTests...)
	tests = append(tests, cgTests...)
	method := &CG{
		Variant:     &DaiYuan{},
		InitialStep: &QuadraticStepSize{},
	}
	testLocal(t, tests, method)
	testLocalFromMinima(t, method)
}

func TestDaiYuanFirstOrderStep(t *testing.T) {
	var tests []unconstrainedTest
	tests = append(tests, gradientDescentTests...)
	tests = append(tests, cgTests...)
	method := &CG{
		Variant:     &DaiYuan{},
		InitialStep: &FirstOrderStepSize{},
	}
	testLocal(t, tests, method)
	testLocalFromMinima(t, method)
}

func TestHagerZhangQuadStep(t *testing.T) {
	var tests []unconstrainedTest
	tests = append(tests, gradientDescentTests...)
	tests = append(tests, cgTests...)
	method := &CG{
		Variant:     &HagerZhang{},
		InitialStep: &QuadraticStepSize{},
	}
	testLocal(t, tests, method)
	testLocalFromMinima(t, method)
}

func TestHagerZhangFirstOrderStep(t *testing.T) {
	var tests []unconstrainedTest
	tests = append(tests, gradientDescentTests...)
	tests = append(tests, cgTests...)
	method := &CG{
		Variant:     &HagerZhang{},
		InitialStep: &FirstOrderStepSize{},
	}
	testLocal(t, tests, method)
	testLocalFromMinima(t, method)
}

func TestBFGS(t *testing.T) {
	var tests []unconstrainedTest
	tests = append(tests, gradientDescentTests...)
	tests = append(tests, newtonTests...)
	tests = append(tests, bfgsTests...)
	method := &BFGS{}
	testLocal(t, tests, method)
	testLocalFromMinima(t, method)
	testLocalFromPerturbedMinima(t, method, 1e-4)
}

func TestLBFGS(t *testing.T) {
	var tests []unconstrainedTest
	tests = append(tests, gradientDescentTests...)
	tests = append(tests, newtonTests...)
	tests = append(tests, lbfgsTests...)
	method := &LBFGS{}
	testLocal(t, tests, method)
	testLocalFromMinima(t, method)
	testLocalFromPerturbedMinima(t, method, 1e-4)
}

func testLocal(t *testing.T, tests []unconstrainedTest, method Method) {
	for _, test := range tests {
		if test.long && testing.Short() {
			continue
		}

		settings := &Settings{
			FunctionAbsTol: math.Inf(-1),
		}
		if test.gradTol == 0 {
			test.gradTol = 1e-12
		}
		settings.GradientAbsTol = test.gradTol

		result, err := Local(test.f, test.x, settings, method)
		if err != nil {
			t.Errorf("error finding minimum (%v) for:\n%v", err, test)
			continue
		}

		if result == nil {
			t.Errorf("nil result without error for:\n%v", test)
			continue
		}

		funcs, funcInfo := getFunctionInfo(test.f)

		// Evaluate the norm of the gradient at the found optimum location.
		var optF, optNorm float64
		if funcInfo.IsFunctionGradient {
			g := make([]float64, len(test.x))
			optF = funcs.gradFunc.FDf(result.X, g)
			optNorm = floats.Norm(g, math.Inf(1))
		} else {
			optF = funcs.function.F(result.X)
			if funcInfo.IsGradient {
				g := make([]float64, len(test.x))
				funcs.gradient.Df(result.X, g)
				optNorm = floats.Norm(g, math.Inf(1))
			}
		}

		// Check that the function value at the found optimum location is
		// equal to result.F
		if optF != result.F {
			t.Errorf("Function value at the optimum location %v not equal to the returned value %v for:\n%v",
				optF, result.F, test)
		}

		// Check that the norm of the gradient at the found optimum location is
		// smaller than the tolerance.
		if optNorm >= settings.GradientAbsTol {
			t.Errorf("Norm of the gradient at the optimum location %v not smaller than tolerance %v for:\n%v",
				optNorm, settings.GradientAbsTol, test)
		}

		// We are going to restart the solution using a fixed starting gradient
		// and value, so evaluate them.
		settings.UseInitialData = true
		if funcInfo.IsFunctionGradient {
			settings.InitialGradient = resize(settings.InitialGradient, len(test.x))
			settings.InitialFunctionValue = funcs.gradFunc.FDf(test.x, settings.InitialGradient)
		} else {
			settings.InitialFunctionValue = funcs.function.F(test.x)
			if funcInfo.IsGradient {
				settings.InitialGradient = resize(settings.InitialGradient, len(test.x))
				funcs.gradient.Df(test.x, settings.InitialGradient)
			}
		}

		// Rerun the test again to make sure that it gets the same answer with
		// the same starting condition. Moreover, we are using the initial data
		// in settings.InitialFunctionValue and settings.InitialGradient.
		result2, err2 := Local(test.f, test.x, settings, method)
		if err2 != nil {
			t.Errorf("error finding minimum second time (%v) for:\n%v", err2, test)
			continue
		}

		if result2 == nil {
			t.Errorf("second time nil result without error for:\n%v", test)
			continue
		}

		// At the moment all the optimizers are deterministic, so check that we
		// get _exactly_ the same answer second time as well.
		if result.F != result2.F {
			t.Errorf("Different minimum second time. First: %v, Second: %v for:\n%v",
				result.F, result2.F, test)
		}

		// Check that providing initial data reduces the number of function
		// and/or gradient calls exactly by one.
		if funcInfo.IsFunctionGradient {
			if result.FunctionGradientEvals != result2.FunctionGradientEvals+1 {
				t.Errorf("Providing initial data does not reduce the number of function/gradient calls for:\n%v", test)
				continue
			}
		} else {
			if result.FunctionEvals != result2.FunctionEvals+1 {
				t.Errorf("Providing initial data does not reduce the number of functions calls for:\n%v", test)
				continue
			}
			if funcInfo.IsGradient {
				if result.GradientEvals != result2.GradientEvals+1 {
					t.Errorf("Providing initial data does not reduce the number of gradient calls for:\n%v", test)
					continue
				}
			}
		}
	}
}

var unconstrainedFuncs = []Function{
	funcs.Beale{},
	funcs.BiggsEXP2{},
	funcs.BiggsEXP3{},
	funcs.BiggsEXP4{},
	funcs.BiggsEXP5{},
	funcs.BiggsEXP6{},
	funcs.Box3D{},
	funcs.BrownBadlyScaled{},
	funcs.BrownAndDennis{},
	funcs.ExtendedPowellSingular{},
	funcs.ExtendedRosenbrock{},
	funcs.Gaussian{},
	funcs.GulfResearchAndDevelopment{},
	funcs.HelicalValley{},
	funcs.Linear{},
	funcs.PenaltyI{},
	funcs.PenaltyII{},
	funcs.PowellBadlyScaled{},
	funcs.Trigonometric{},
	funcs.VariablyDimensioned{},
	funcs.Watson{},
	funcs.Wood{},
}

// testLocalFromMinima tests if Local() can find the minimum with method when
// the starting point is the minimum itself.
// This test will probably have to be revised once derivative-free methods are
// introduced or not used for such methods.
func testLocalFromMinima(t *testing.T, method Method) {
	// Keep this here until functions.minimumer is exported.
	type minimumer interface {
		Minima() []funcs.Minimum
	}

	for _, f := range unconstrainedFuncs {
		minimumer, isMinimumer := f.(minimumer)
		if !isMinimumer {
			continue
		}

		settings := &Settings{
			FunctionAbsTol: math.Inf(-1),
			GradientAbsTol: 1e-9, // Gradients at all the minima have their inf norm smaller than this value...
		}
		// ... except for BrownAndDennis whose minimum is known with much less accuracy.
		if reflect.TypeOf(f) == reflect.TypeOf(funcs.BrownAndDennis{}) {
			settings.GradientAbsTol = 1e-5
		}

		for i, min := range minimumer.Minima() {
			// Try starting the optimizer from an optimum location given by min.X.
			result, err := Local(f, min.X, settings, method)
			if err != nil {
				t.Errorf("%v, minimum #%d: error finding minimum from an optimum location (%v)",
					reflect.TypeOf(f), i, err)
			}
			// Minimization should not have started at all because all minima
			// are known with at least the above accuracy.
			if result.MajorIterations > 0 {
				t.Errorf("%v, minimum #%d: too many iterations", reflect.TypeOf(f), i)
			}
		}
	}
}

// testLocalFromPerturbedMinima tests the ability of method to find a minimum
// when the starting point is located in the neighborhood of the minimum. Size
// of the neighborhood is determined by scale.
// Note that this test is very difficult to pass.
func testLocalFromPerturbedMinima(t *testing.T, method Method, scale float64) {
	// Keep this here until functions.minimumer is exported.
	type minimumer interface {
		Minima() []funcs.Minimum
	}

	for _, f := range unconstrainedFuncs {
		minimumer, isMinimumer := f.(minimumer)
		if !isMinimumer {
			continue
		}
		// Skip BrownAndDennis as it is notoriously difficult to minimize, even
		// more so when starting in the neighborhood of its minimum.
		if reflect.TypeOf(f) == reflect.TypeOf(funcs.BrownAndDennis{}) {
			continue
		}

		settings := &Settings{
			FunctionAbsTol: math.Inf(-1),
			GradientAbsTol: 1e-6,
		}
		for i, min := range minimumer.Minima() {
			// Perturb the minimum and try starting the minimizer from there.
			for j, v := range min.X {
				// TODO(vladimir-ch): What is the best way to perturb the minimum for testing?
				// c is uniformly distributed in [-scale, scale)
				// c := scale * (2*rand.Float64() - 1)
				// c is one of {-scale, 0, scale}
				c := scale * float64(rand.Intn(3)-1)
				min.X[j] = (1 + c) * v
			}
			_, err2 := Local(f, min.X, settings, method)
			if err2 != nil {
				t.Errorf("%v, minimum #%d: error finding minimum from randomly perturbed optimum location (%v)",
					reflect.TypeOf(f), i, err2)
			}
		}
	}
}
