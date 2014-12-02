// Copyright ©2014 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package optimize

import "github.com/gonum/floats"

// GradientDescent is a Method that performs gradient-based optimization. Gradient
// Descent performs successive steps along the direction of the gradient. The
// LinesearchMethod specifies the kind of linesearch to be done, and StepSizer determines
// the initial step size of each direction. If either LinesearchMethod or StepSizer
// are nil, a reasonable value will be chosen.
type GradientDescent struct {
	LinesearchMethod LinesearchMethod
	InitialStep      StepSizer

	linesearch *Linesearch
}

func (g *GradientDescent) Init(l Location, f *FunctionInfo, xNext []float64) (EvaluationType, IterationType, error) {
	if g.LinesearchMethod == nil {
		g.LinesearchMethod = &Backtracking{}
	}
	if g.InitialStep == nil {
		g.InitialStep = &QuadraticStepSize{}
	}
	if g.linesearch == nil {
		g.linesearch = &Linesearch{}
	}
	g.linesearch.Method = g.LinesearchMethod
	g.linesearch.InitialStep = g.InitialStep
	g.linesearch.NextDirectioner = g

	return g.linesearch.Init(l, f, xNext)
}

func (g *GradientDescent) Iterate(l Location, xNext []float64) (EvaluationType, IterationType, error) {
	return g.linesearch.Iterate(l, xNext)
}

func (g *GradientDescent) InitDirection(l Location, direction []float64) {
	copy(direction, l.Gradient)
	floats.Scale(-1, direction)
}

func (g *GradientDescent) NextDirection(l Location, direction []float64) {
	copy(direction, l.Gradient)
	floats.Scale(-1, direction)
}
