package lp

import (
	"math"

	"github.com/gonum/blas"
	"github.com/gonum/blas/blas64"
	"github.com/gonum/matrix/mat64"
)

// Modified cholesky decomposition for use in interior point algorithms.
type ModChol struct {
	data    []float64
	stride  int
	uplo    blas.Uplo
	skipped []int
	n       int
}

// Modifies the given symmetric matrix in place.
func (m *ModChol) From(A blas64.Symmetric, eps float64) {
	ul := A.Uplo
	n := A.N
	a := A.Data
	lda := A.Stride

	bi := blas64.Implementation()
	if ul != blas.Upper && ul != blas.Lower {
		panic("bad uplo")
	}
	if n < 0 {
		panic("n < 0")
	}
	if lda < n {
		panic("bad stride")
	}
	if n == 0 {
		*m = ModChol{
			uplo:   ul,
			stride: lda,
			data:   a,
			n:      n,
		}
		return
	}

	beta := 0.0
	for j := 0; j < n; j++ {
		ajj := a[j+j*lda]
		if ajj > beta {
			beta = ajj
		}
	}
	if beta <= 0.0 {
		beta = 1
	}

	var skipped []int

	if ul == blas.Upper {
		for j := 0; j < n; j++ {
			ajj := a[j+j*lda]
			if ajj < beta*eps {
				// skip step
				skipped = append(skipped, j)
				continue
			}
			ajj = math.Sqrt(ajj)
			a[j+j*lda] = ajj
			if j < n-1 {
				bi.Dscal(n-j-1, 1/ajj, a[j*lda+j+1:], 1)
				bi.Dsyr(blas.Upper, n-j-1, -1, a[j*lda+(j+1):], 1, a[(j+1)*lda+(j+1):], lda)
			}
		}
		*m = ModChol{
			uplo:    ul,
			stride:  lda,
			skipped: skipped,
			data:    a,
			n:       n,
		}
		return
	}
	for j := 0; j < n; j++ {
		for j := 0; j < n; j++ {
			ajj := a[j+j*lda]
			if ajj < beta*eps {
				// skip step
				skipped = append(skipped, j)
				continue
			}
			ajj = math.Sqrt(ajj)
			a[j+j*lda] = ajj
			if j < n-1 {
				bi.Dscal(n-j-1, 1/ajj, a[(j+1)*lda+j:], lda)
				bi.Dsyr(blas.Upper, n-j-1, -1, a[(j+1)*lda+j:], lda, a[j+1+(j+1)*lda:], lda)
			}
		}
		*m = ModChol{
			uplo:    ul,
			stride:  lda,
			skipped: skipped,
			data:    a,
			n:       n,
		}
	}
	return
}

func SolveModChol(m *ModChol, x *mat64.Vector, y *mat64.Vector) {
	if x.Len() != m.n || y.Len() != m.n {
		panic("SolveModChol: dimension mismatch")
	}

	xdat := x.RawVector().Data
	ydat := y.RawVector().Data

	lda := m.stride
	bi := blas64.Implementation()
	if m.uplo == blas.Upper {
		skipped := m.skipped
		for j := 0; j < m.n; j++ {
			if len(skipped) > 0 && j == skipped[0] {
				ydat[j] = 0
				skipped = skipped[1:]
				continue
			}
			ydat[j] = (xdat[j] - bi.Ddot(j, ydat, 1, m.data[j:], lda)) / m.data[j+j*lda]
		}
		skipped = m.skipped
		for j := m.n - 1; j >= 0; j-- {
			if len(skipped) > 0 && j == skipped[len(skipped)-1] {
				skipped = skipped[:len(skipped)-1]
				continue
			}
			sub := 0.0
			if j < m.n-1 {
				sub = bi.Ddot(m.n-j-1, ydat[j+1:], 1, m.data[j*lda+(j+1):], 1)
			}
			ydat[j] = (ydat[j] - sub) / m.data[j+j*lda]
		}
		return
	}

	skipped := m.skipped
	for j := 0; j < m.n; j++ {
		if len(skipped) > 0 && j == skipped[0] {
			skipped = skipped[1:]
			continue
		}
		ydat[j] = (xdat[j] - bi.Ddot(j, ydat, 1, m.data[j:], lda)) / m.data[j+j*lda]
	}
	skipped = m.skipped
	for j := m.n - 1; j >= 0; j-- {
		if len(skipped) > 0 && j == skipped[len(skipped)-1] {
			skipped = skipped[:len(skipped)-1]
			continue
		}
		ydat[j] = (ydat[j] - bi.Ddot(m.n-j-1, ydat[j+1:], 1, m.data[j+1+j*lda:], 1)) / m.data[j+j*lda]
	}
}
