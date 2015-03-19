// Copyright ©2015 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package optimize

import "math"

// CoordinateDescent optimizes a function through successive line-searches along
// a single coordinate direction.
type CoordinateDescent struct {
	LinesearchMethod LinesearchMethod
	InitialStep      StepSizer

	linesearch *Linesearch
	coord      int
	up         bool
}

func (*CoordinateDescent) Needs() struct {
	Gradient bool
	Hessian  bool
} {
	return struct {
		Gradient bool
		Hessian  bool
	}{true, false}
}

func (c *CoordinateDescent) Init(loc *Location, f *FunctionInfo, xNext []float64) (EvaluationType, IterationType, error) {
	if c.LinesearchMethod == nil {
		c.LinesearchMethod = &Bisection{}
	}
	if c.linesearch == nil {
		c.linesearch = &Linesearch{}
	}
	if c.InitialStep == nil {
		c.InitialStep = &FirstOrderStepSize{}
	}
	c.linesearch.Method = c.LinesearchMethod
	c.linesearch.NextDirectioner = c

	return c.linesearch.Init(loc, f, xNext)
}

func (c *CoordinateDescent) Iterate(loc *Location, xNext []float64) (EvaluationType, IterationType, error) {
	return c.linesearch.Iterate(loc, xNext)
}

func (c *CoordinateDescent) InitDirection(loc *Location, dir []float64) (stepSize float64) {
	for i := range dir {
		dir[i] = 0
	}
	c.coord = -1
	var max float64
	for i, v := range loc.Gradient {
		if math.Abs(v) > max {
			c.coord = i
			max = math.Abs(v)
		}
	}
	dir[c.coord] = 1
	if loc.Gradient[c.coord] > 0 {
		dir[c.coord] = -1
	}
	return c.InitialStep.Init(loc, dir)
}

func (c *CoordinateDescent) NextDirection(loc *Location, dir []float64) (stepSize float64) {
	// Set the next step direction as the coordinate with the steepest descent
	// that is not the previous direction.
	idx := -1
	var max float64
	for i, v := range loc.Gradient {
		if i == c.coord {
			continue
		}
		if math.Abs(v) > max {
			idx = i
			max = math.Abs(v)
		}
	}
	if len(loc.X) == 1 {
		c.coord = 0
	}
	c.coord = idx
	for i := range dir {
		dir[i] = 0
	}
	dir[c.coord] = 1
	if loc.Gradient[c.coord] > 0 {
		dir[c.coord] = -1
	}
	return c.InitialStep.StepSize(loc, dir)
}
