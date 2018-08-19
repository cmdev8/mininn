package main

import "gonum.org/v1/gonum/mat"

func (nn *nn) train(ts trainingSettings, x, y *mat.Dense) error {
	output := new(mat.Dense)

	for i := 0; i < ts.epochs; i++ {
		if err := nn.backpropagate(ts, x, y, output); err != nil {
			return err
		}
	}

	return nil
}

func (nn *nn) backpropagate(ts trainingSettings, x, y, output *mat.Dense) error {
	l0 := new(mat.Dense)
	l0.Mul(x, nn.Layers[0].W)
	l0.Apply(func(_, col int, v float64) float64 { return v + nn.Layers[0].B.At(0, col) }, l0)

	l0activation := new(mat.Dense)
	l0activation.Apply(ts.activationFunc, l0)

	l1 := new(mat.Dense)
	l1.Mul(l0activation, nn.Layers[1].W)
	l1.Apply(func(_, col int, v float64) float64 { return v + nn.Layers[1].B.At(0, col) }, l1)
	output.Apply(ts.activationFunc, l1)

	networkError := new(mat.Dense)
	networkError.Sub(y, output)

	s0 := new(mat.Dense)
	s0.Apply(ts.activationFuncDerivative, output)
	s1 := new(mat.Dense)
	s1.Apply(ts.activationFuncDerivative, l0activation)

	do := new(mat.Dense)
	do.MulElem(networkError, s0)
	error0 := new(mat.Dense)
	error0.Mul(do, nn.Layers[1].W.T())

	dh := new(mat.Dense)
	dh.MulElem(error0, s1)

	woa := new(mat.Dense)
	woa.Mul(l0activation.T(), do)
	woa.Scale(ts.lr, woa)
	nn.Layers[1].W.Add(nn.Layers[1].W, woa)

	boa, err := sumAxis(0, do)
	if err != nil {
		return err
	}
	boa.Scale(ts.lr, boa)
	nn.Layers[1].B.Add(nn.Layers[1].B, boa)

	wha := new(mat.Dense)
	wha.Mul(x.T(), dh)
	wha.Scale(ts.lr, wha)
	nn.Layers[0].W.Add(nn.Layers[0].W, wha)

	bha, err := sumAxis(0, dh)
	if err != nil {
		return err
	}

	bha.Scale(ts.lr, bha)
	nn.Layers[0].B.Add(nn.Layers[0].B, bha)

	return nil
}
