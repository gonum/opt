// Copyright ©2014 The gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package optimize

import (
	"math"
	"time"

	"github.com/gonum/floats"
)

// Local finds a local minimum of a function using a sequential algorithm.
// In order to maximize a function, multiply the output by -1.
//
// The first argument is of Function type representing the function to be minimized.
// Type switching is used to see if the function implements Gradient, FunctionGradient
// and Statuser.
//
// The second argument is the initial location at which to start the minimization.
// The initial location must be supplied, and must have a length equal to the
// problem dimension.
//
// The third argument contains the settings for the minimization. It is here that
// gradient tolerance, etc. are specified. The DefaultSettings() function
// can be called for a Settings struct with the default values initialized.
// If settings == nil, the default settings are used. Please see the documentation
// for the Settings structure for more information. The optimization Method used
// may also contain settings, see documentation for the appropriate optimizer.
//
// The final argument is the optimization method to use. If method == nil, then
// an appropriate default is chosen based on the properties of the other arguments
// (dimension, gradient-free or gradient-based, etc.). The optimization
// methods in this package are designed such that reasonable defaults occur
// if options are not specified explicitly. For example, the code
//  method := &Bfgs{}
// creates a pointer to a new Bfgs struct. When minimize is called, the settings
// in the method will be populated with default values. The methods are also
// designed such that they can be reused in future calls to method.
//
// Local returns a Result struct and any error that occurred. Please see the
// documentation of Result for more information.
//
// Please be aware that the default behavior of Local is to find the minimum.
// For certain functions and optimization methods, this process can take many
// function evaluations. If you would like to put limits on this, for example
// maximum runtime or maximum function evaluations, please modify the Settings
// input struct.
func Local(f Function, initX []float64, settings *Settings, method Method) (*Result, error) {
	if len(initX) == 0 {
		panic("local: initial X has zero length")
	}

	startTime := time.Now()

	funcs, funcInfo := getFunctionInfo(f)

	if method == nil {
		method = getDefaultMethod(funcInfo)
	}

	if settings == nil {
		settings = DefaultSettings()
	}

	stats := &Stats{}
	loc, err := getStartingLocation(f, funcs, funcInfo, initX, stats, settings)
	if err != nil {
		return nil, err
	}

	// update stats (grad norm, function value, etc.) so that things are
	// initialized for the first convergence check
	update(stats, funcInfo, NoEvaluation, NoIteration, startTime)

	if settings.Recorder != nil {
		err = settings.Recorder.Init(funcInfo)
		if err != nil {
			return &Result{Status: Failure}, err
		}
	}

	// actually perform the minimization
	status, err := minimize(settings, &loc, method, funcInfo, stats, funcs, startTime)

	var optLoc Location
	copyLocation(&optLoc, loc)

	// cleanup at exit
	if settings.Recorder != nil && err == nil {
		err = settings.Recorder.Record(optLoc, NoEvaluation, PostIteration, stats)
	}
	stats.Runtime = time.Since(startTime)
	return &Result{
		Stats:    *stats,
		Location: optLoc,
		Status:   status,
	}, err
}

func minimize(settings *Settings, loc *Location, method Method, funcInfo *FunctionInfo, stats *Stats, funcs functions, startTime time.Time) (status Status, err error) {
	methodStatus, methodIsStatuser := method.(Statuser)
	xNext := make([]float64, len(loc.X))

	evalType, iterType, err := method.Init(*loc, funcInfo, xNext)
	if err != nil {
		return Failure, err
	}

	for {
		if settings.Recorder != nil {
			err = settings.Recorder.Record(*loc, evalType, iterType, stats)
			if err != nil {
				status = Failure
				return
			}
		}

		status = checkConvergence(*loc, iterType, stats, settings)
		if status != NotTerminated {
			return
		}

		if funcInfo.IsStatuser {
			status, err = funcs.status.Status()
			if err != nil || status != NotTerminated {
				return
			}
		}

		if methodIsStatuser {
			status, err = methodStatus.Status()
			if err != nil || status != NotTerminated {
				return
			}
		}

		// Compute the new function and update the statistics
		err = evaluate(funcs, funcInfo, evalType, xNext, loc, stats)
		if err != nil {
			status = Failure
			return
		}
		update(stats, funcInfo, evalType, iterType, startTime)

		// Find the next location
		evalType, iterType, err = method.Iterate(*loc, xNext)
		if err != nil {
			status = Failure
			return
		}
	}
	panic("unreachable")
}

func copyLocation(dst *Location, src Location) {
	dst.X = resize(dst.X, len(src.X))
	copy(dst.X, src.X)

	dst.F = src.F

	dst.Gradient = resize(dst.Gradient, len(src.Gradient))
	copy(dst.Gradient, src.Gradient)
}

func getFunctionInfo(f Function) (functions, *FunctionInfo) {
	// Not sure how/if we want to compute timing to be used with functions
	gradient, isGradient := f.(Gradient)
	gradFunc, isFunGrad := f.(FunctionGradient)

	status, isStatuser := f.(Statuser)

	funcInfo := &FunctionInfo{
		IsGradient:         isGradient,
		IsFunctionGradient: isFunGrad,
		IsStatuser:         isStatuser,
	}
	funcs := functions{
		function: f,
		gradient: gradient,
		gradFunc: gradFunc,
		status:   status,
	}

	return funcs, funcInfo
}

func getDefaultMethod(funcInfo *FunctionInfo) Method {
	if funcInfo.IsGradient || funcInfo.IsFunctionGradient {
		return &BFGS{}
	}
	// TODO: Implement a gradient-free method
	panic("optimize: gradient-free methods not yet coded")
}

// Combine location and stats because maybe in the future we'll add evaluation times
// to functionStats?
func getStartingLocation(f Function, funcs functions, funcInfo *FunctionInfo, initX []float64, stats *Stats, settings *Settings) (Location, error) {
	dim := len(initX)
	loc := Location{
		X: make([]float64, dim),
	}
	copy(loc.X, initX)

	if settings.UseInitialData {
		loc.F = settings.InitialFunctionValue
		if funcInfo.IsGradient || funcInfo.IsFunctionGradient {
			initG := settings.InitialGradient
			if len(initG) != dim {
				panic("local: initial location size mismatch")
			}
			loc.Gradient = make([]float64, dim)
			copy(loc.Gradient, initG)
			stats.GradientNorm = floats.Norm(loc.Gradient, math.Inf(1))
		}
	} else {
		// Compute missing information in the initial state.
		if funcInfo.IsGradient || funcInfo.IsFunctionGradient {
			loc.Gradient = make([]float64, dim)
			evaluate(funcs, funcInfo, FunctionAndGradientEval, loc.X, &loc, stats)
		} else {
			evaluate(funcs, funcInfo, FunctionEval, loc.X, &loc, stats)
		}
	}

	if math.IsNaN(loc.F) {
		return loc, ErrNaN
	}
	if math.IsInf(loc.F, 1) {
		return loc, ErrInf
	}
	return loc, nil
}

func checkConvergence(loc Location, itertype IterationType, stats *Stats, settings *Settings) Status {
	if itertype == MajorIteration && loc.Gradient != nil {
		if stats.GradientNorm <= settings.GradientAbsTol {
			return GradientAbsoluteConvergence
		}
	}

	if itertype == MajorIteration && loc.F < settings.FunctionAbsTol {
		return FunctionAbsoluteConvergence
	}

	// Check every step for negative infinity because it could break the
	// linesearches and -inf is the best you can do anyway.
	if math.IsInf(loc.F, -1) {
		return FunctionNegativeInfinity
	}

	if settings.FunctionEvals > 0 {
		totalFun := stats.FunctionEvals + stats.FunctionGradientEvals
		if totalFun >= settings.FunctionEvals {
			return FunctionEvaluationLimit
		}
	}

	if settings.GradientEvals > 0 {
		totalGrad := stats.GradientEvals + stats.FunctionGradientEvals
		if totalGrad >= settings.GradientEvals {
			return GradientEvaluationLimit
		}
	}

	if settings.Runtime > 0 {
		if stats.Runtime >= settings.Runtime {
			return RuntimeLimit
		}
	}

	if itertype == MajorIteration && settings.MajorIterations > 0 {
		if stats.MajorIterations >= settings.MajorIterations {
			return IterationLimit
		}
	}
	return NotTerminated
}

// evaluate evaluates the function and stores the answer in place
func evaluate(funcs functions, funcInfo *FunctionInfo, evalType EvaluationType, xNext []float64, loc *Location, stats *Stats) error {
	sameX := floats.Equal(loc.X, xNext)
	if !sameX {
		copy(loc.X, xNext)
	}
	switch evalType {
	case FunctionEval:
		loc.F = funcs.function.F(loc.X)
		stats.FunctionEvals++
		if !sameX {
			for i := range loc.Gradient {
				loc.Gradient[i] = math.NaN()
			}
			stats.GradientNorm = math.NaN()
		}
		return nil
	case GradientEval:
		if funcInfo.IsGradient {
			if !sameX {
				loc.F = math.NaN()
			}
			funcs.gradient.Df(loc.X, loc.Gradient)
			stats.GradientEvals++
			stats.GradientNorm = floats.Norm(loc.Gradient, math.Inf(1))
			return nil
		}
		if funcInfo.IsFunctionGradient {
			loc.F = funcs.gradFunc.FDf(loc.X, loc.Gradient)
			stats.FunctionGradientEvals++
			stats.GradientNorm = floats.Norm(loc.Gradient, math.Inf(1))
			return nil
		}
		return ErrMismatch{Type: evalType}
	case FunctionAndGradientEval:
		if funcInfo.IsFunctionGradient {
			loc.F = funcs.gradFunc.FDf(loc.X, loc.Gradient)
			stats.FunctionGradientEvals++
			stats.GradientNorm = floats.Norm(loc.Gradient, math.Inf(1))
			return nil
		}
		if funcInfo.IsGradient {
			loc.F = funcs.function.F(loc.X)
			stats.FunctionEvals++
			funcs.gradient.Df(loc.X, loc.Gradient)
			stats.GradientEvals++
			stats.GradientNorm = floats.Norm(loc.Gradient, math.Inf(1))
			return nil
		}
		return ErrMismatch{Type: evalType}
	default:
		panic("unreachable")
	}
}

// update updates the stats given the new evaluation
func update(stats *Stats, funcInfo *FunctionInfo, evalType EvaluationType, iterType IterationType, startTime time.Time) {
	if iterType == MajorIteration {
		stats.MajorIterations++
	}
	stats.Runtime = time.Since(startTime)
}
