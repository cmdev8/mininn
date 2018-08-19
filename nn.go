package main

func newNN(l layout) *nn {
	return &nn{
		Layout: l,
		Layers: []layer{
			layer{W: initRandMatrix(l.NFeatures, l.NHidden), B: initRandMatrix(1, l.NHidden)},
			layer{W: initRandMatrix(l.NHidden, l.NClasses), B: initRandMatrix(1, l.NClasses)},
		},
	}
}
