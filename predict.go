package main

import (
	"gonum.org/v1/gonum/mat"
)

func (nn *nn) predict(x *mat.Dense, ts trainingSettings) (*mat.Dense, error) {
	output := new(mat.Dense)

	l0 := new(mat.Dense)
	l0.Mul(x, nn.Layers[0].W)
	l0.Apply(func(_, col int, v float64) float64 { return v + nn.Layers[0].B.At(0, col) }, l0)

	l0activations := new(mat.Dense)
	l0activations.Apply(ts.activationFunc, l0)

	l1 := new(mat.Dense)
	l1.Mul(l0activations, nn.Layers[1].W)
	l1.Apply(func(_, col int, v float64) float64 { return v + nn.Layers[1].B.At(0, col) }, l1)
	output.Apply(ts.activationFunc, l1)

	return output, nil
}
