package main

import (
	"log"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

func accuracy(nn *nn, testData *mat.Dense, testTarget *mat.Dense, ts trainingSettings) float64 {
	predictions, err := nn.predict(testData, ts)
	if err != nil {
		log.Fatal(err)
	}

	var t int
	n, _ := predictions.Dims()
	for i := 0; i < n; i++ {

		labelRow := mat.Row(nil, i, testTarget)
		var prediction int
		for idx, label := range labelRow {
			if label == 1.0 {
				prediction = idx
				break
			}
		}
		if predictions.At(i, prediction) == floats.Max(mat.Row(nil, i, predictions)) {
			t++
		}
	}

	return float64(t) / float64(n)
}
