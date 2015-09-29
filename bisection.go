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

	initF float64
	initG float64

	minStep float64
	minF    float64
	minG    float64

	maxStep float64
	maxF    float64
	maxG    float64

	step float64
	f    float64

	lastOp Operation
}

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
	b.initG = g

	b.minStep = 0
	b.minF = f
	b.minG = g

	b.maxStep = math.Inf(1)
	b.maxF = math.NaN()
	b.maxG = math.NaN()

	b.step = step
	b.f = math.NaN()

	b.lastOp = FuncEvaluation
	return b.lastOp
}

func (b *Bisection) Iterate(f, g float64) (Operation, float64, error) {
	if b.lastOp&(FuncEvaluation|GradEvaluation) == 0 {
		panic("bisection: Init has not been called")
	}

	if b.lastOp == FuncEvaluation {
		b.f = f
		b.lastOp = GradEvaluation
		return b.lastOp, b.step, nil
	}
	f = b.f

	// Don't finish the linesearch until a minimum is found that is better than
	// the best point found so far. We want to end up in the lowest basin of
	// attraction
	minF := b.initF
	if b.maxF < minF {
		minF = b.maxF
	}
	if b.minF < minF {
		minF = b.minF
	}
	if StrongWolfeConditionsMet(f, g, minF, b.initG, b.step, 0, b.GradConst) {
		b.lastOp = MajorIteration
		return b.lastOp, b.step, nil
	}

	// Deciding on the next step size
	if math.IsInf(b.maxStep, 1) {
		// Have not yet bounded the minimum
		switch {
		case g > 0:
			// Found a change in derivative sign, so this is the new maximum
			b.maxStep = b.step
			b.maxF = f
			b.maxG = g
			return b.nextStep((b.minStep + b.maxStep) / 2)
		case f <= b.minF:
			// Still haven't found an upper bound, but there is not an increase in
			// function value and the gradient is still negative, so go more in
			// that direction.
			b.minStep = b.step
			b.minF = f
			b.minG = g
			return b.nextStep(b.step * 2)
		default:
			// Increase in function value, but the gradient is still negative.
			// Means we must have skipped over a local minimum, so set this point
			// as the new maximum
			b.maxStep = b.step
			b.maxF = f
			b.maxG = g
			return b.nextStep((b.minStep + b.maxStep) / 2)
		}
	}

	// Already bounded the minimum, but wolfe conditions not met. Need to step to
	// find minimum.
	if f <= b.minF && f <= b.maxF {
		if g < 0 {
			b.minStep = b.step
			b.minF = f
			b.minG = g
		} else {
			b.maxStep = b.step
			b.maxF = f
			b.maxG = g
		}
	} else {
		// We found a higher point. Want to push toward the minimal bound
		if b.minF <= b.maxF {
			b.maxStep = b.step
			b.maxF = f
			b.maxG = g
		} else {
			b.minStep = b.step
			b.minF = f
			b.minG = g
		}
	}
	return b.nextStep((b.minStep + b.maxStep) / 2)
}

// nextStep checks if the new step is equal to the old step.
// This can happen if min and max are the same, or if the step size is infinity,
// both of which indicate the minimization must stop. If the steps are different,
// it sets the new step size and returns the evaluation type and the step. If the steps
// are the same, it returns an error.
func (b *Bisection) nextStep(step float64) (Operation, float64, error) {
	if b.step == step {
		b.lastOp = NoOperation
		return b.lastOp, b.step, ErrLinesearcherFailure
	}
	b.step = step
	b.lastOp = FuncEvaluation
	return b.lastOp, b.step, nil
}
