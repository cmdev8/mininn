package main

import (
	"flag"
	"fmt"
	"log"

	"gonum.org/v1/gonum/mat"
)

type nn struct {
	Layout layout
	Layers []layer
}

type layer struct {
	W *mat.Dense
	B *mat.Dense
}

type layout struct {
	NFeatures int
	NClasses  int
	NHidden   int
}

type trainingSettings struct {
	lr                       float64
	epochs                   int
	activationFunc           func(_, _ int, v float64) float64
	activationFuncDerivative func(_, _ int, v float64) float64
}

type dataset struct {
	features []float64
	lables   []float64
}

func main() {
	trainingMode := flag.Bool("train", false, "Training mode")
	predictionMode := flag.Bool("predict", false, "Prediction mode")
	modelPath := flag.String("modelPath", "model.bin", "ModelPath")
	dataset := flag.String("dataset", "", "Dataset csv path")
	nFeatures := flag.Int("n-features", 0, "Number of features")
	nClasses := flag.Int("n-classes", 0, "Number of classes")
	nHidden := flag.Int("n-hidden", 0, "Number of hidden nodes")
	learningRate := flag.Float64("learning-rate", 0.01, "Learning rate")
	nEpochs := flag.Int("n-epochs", 1000, "Number of epochs")

	flag.Parse()

	netLayout := layout{
		NFeatures: *nFeatures,
		NClasses:  *nClasses,
		NHidden:   *nHidden,
	}

	trainingSettings := trainingSettings{
		lr:                       *learningRate,
		epochs:                   *nEpochs,
		activationFunc:           func(_, _ int, v float64) float64 { return sigmoid(v) },
		activationFuncDerivative: func(_, _ int, v float64) float64 { return sigmoidDerivative(v) },
	}

	if *trainingMode {
		training(netLayout, *modelPath, *dataset, trainingSettings)
	}

	if *predictionMode {
		prediction(netLayout, *modelPath, *dataset, trainingSettings)
	}
}

func training(netLayout layout, outPath string, datasetPath string, trainingSettings trainingSettings) {
	fmt.Println("Training")

	trainingFeatures, trainingTarget, testFeatures, testTargets := loadTrainingDataset(datasetPath, netLayout, 0.3)

	network := newNN(netLayout)
	if err := network.train(trainingSettings, trainingFeatures, trainingTarget); err != nil {
		log.Fatal(err)
	}

	saveErr := saveObject(outPath, network)
	if saveErr != nil {
		log.Fatal(saveErr)
	}

	fmt.Printf("\nAccuracy = %0.2f\n\n", accuracy(network, testFeatures, testTargets, trainingSettings))
}

func prediction(netLayout layout, modelPath string, datasetPath string, trainingSettings trainingSettings) {
	fmt.Println("Prediction")

	network := new(nn)
	err := loadObject(modelPath, network)
	if err != nil {
		log.Fatal(err)
	}

	predictionFeatures := loadPredictionDataset(datasetPath, netLayout)
	predictions, predictionErr := network.predict(predictionFeatures, trainingSettings)

	if predictionErr != nil {
		log.Fatal(predictionErr)
	}

	n, _ := predictions.Dims()
	for i := 0; i < n; i++ {
		fmt.Println(mat.Row(nil, i, predictions))
	}

}
