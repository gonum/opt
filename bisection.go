// Copyright ©2014 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package optimize

import "math"

// Bisection is a Linesearcher that uses a bisection to find a point that
// satisfies the strong Wolfe conditions with the given gradient constant and
// function constant of zero. If GradConst is zero, it will be set to a reasonable
// value. Bisection will panic if GradConst is not between zero and one.
type Bisection struct {
	GradConst float64

	initF     float64
	initAbsG  float64
	step      [3]float64
	f         [3]float64
	k         int
	bracketed bool
	lastOp    Operation
}

const golden float64 = 1.61803398875

func (b *Bisection) Init(f, g float64, step float64) Operation {
	if step <= 0 {
		panic("bisection: bad step size")
	}
	if g >= 0 {
		panic("bisection: initial derivative is non-negative")
	}

	if b.GradConst == 0 {
		b.GradConst = 0.9
	}
	if b.GradConst <= 0 || b.GradConst >= 1 {
		panic("bisection: GradConst not between 0 and 1")
	}

	b.initF = f
	b.initAbsG = math.Abs(g)
	b.step = [3]float64{0, step, math.NaN()}
	b.f = [3]float64{f, math.NaN(), math.NaN()}
	b.k = 1
	b.bracketed = false

	b.lastOp = FuncEvaluation
	return b.lastOp
}

func (b *Bisection) Iterate(f, g float64) (Operation, float64, error) {
	if b.lastOp&(FuncEvaluation|GradEvaluation) == 0 {
		panic("bisection: Init has not been called")
	}

	k := b.k
	// Make sure that f is up-to-date.
	if b.lastOp == FuncEvaluation {
		b.f[k] = f
	}

	// Don't finish the linesearch until a minimum is found that is better than
	// the best point found so far. We want to end up in the lowest basin of
	// attraction
	minF := b.f[0]
	if b.f[1] < minF {
		minF = b.f[1]
	}
	if b.f[2] < minF {
		minF = b.f[2]
	}
	if b.f[k] <= minF {
		// The step satisfies the Armijo condition.
		// Request the derivative, if necessary, and check the curvature
		// condition.
		if b.lastOp != GradEvaluation {
			b.lastOp = GradEvaluation
			return b.lastOp, b.step[k], nil
		}
		if math.Abs(g) < b.GradConst*b.initAbsG {
			b.lastOp = MajorIteration
			return b.lastOp, b.step[k], nil
		}
	}

	// Continue the search because the step is not satisfactory.

	if !b.bracketed {
		// We have bracketed a local minimum whenever the downhill trend has
		// stopped either because the function value has increased or because
		// the derivative (if available) is positive.
		upF := b.f[k] > b.f[k-1]
		upG := b.lastOp == GradEvaluation && g > 0
		if upF || upG {
			b.bracketed = true
			switch {
			case k == 1: // A minimum is between step[0] and step[1].
				b.step[2] = b.step[1]
				b.f[2] = b.f[1]
			case upG: // A minimum is between step[1] and step[2].
				b.step[0] = b.step[1]
				b.f[0] = b.f[1]
			default: // A minimum is between step[0] and step[2].
			}
			// Invalidate the middle element.
			b.step[1] = math.NaN()
			b.f[1] = math.NaN()
			// From now on store step and f in the middle element.
			b.k = 1
			return b.nextStep()
		}
		// No minimum has been bracketed yet. Slide forward discarding
		// the left-most step.
		if k == 1 {
			b.k = 2
			b.step[2] = (1 + golden) * b.step[1]
		} else {
			s2 := b.step[2]
			b.step = [3]float64{b.step[1], s2, s2 + golden*(s2-b.step[1])}
			b.f = [3]float64{b.f[1], b.f[2], math.NaN()}
		}
		b.lastOp = FuncEvaluation
		return b.lastOp, b.step[2], nil
	}

	// Already bracketed the minimum, but Wolfe conditions are still not met.

	if b.f[1] <= b.f[0] && b.f[1] <= b.f[2] {
		// Value at midpoint is smaller than at either endpoint.
		// We need the derivative to decide where to go.
		if b.lastOp != GradEvaluation {
			b.lastOp = GradEvaluation
			return b.lastOp, b.step[1], nil
		}
		if g < 0 {
			b.step[0] = b.step[1]
			b.f[0] = b.f[1]
		} else {
			b.step[2] = b.step[1]
			b.f[2] = b.f[1]
		}
	} else {
		// We found a higher point. Push toward the minimal bound.
		if b.f[0] <= b.f[2] {
			b.step[2] = b.step[1]
			b.f[2] = b.f[1]
		} else {
			b.step[0] = b.step[1]
			b.f[0] = b.f[1]
		}
	}
	return b.nextStep()
}

// nextStep computes the new step and checks if it is equal to the old one.
// This can happen if min and max are the same, or if the step size is
// infinity, both of which indicate the minimization must stop.
// If the steps are different, it sets the new step size and returns the
// evaluation type and the step.
// If the steps are the same, it returns an error.
func (b *Bisection) nextStep() (Operation, float64, error) {
	step := (b.step[0] + b.step[2]) / 2
	if b.step[1] == step {
		b.lastOp = NoOperation
		return b.lastOp, b.step[1], ErrLinesearcherFailure
	}
	b.step[1] = step
	b.lastOp = FuncEvaluation
	return b.lastOp, b.step[1], nil
}
