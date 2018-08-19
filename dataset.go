package main

import (
	"encoding/csv"
	"log"
	"os"
	"strconv"

	"gonum.org/v1/gonum/mat"
)

func readDatasetCSV(fileName string, nFields int) [][]string {
	f, err := os.Open(fileName)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	reader := csv.NewReader(f)
	reader.FieldsPerRecord = nFields

	rawCSVData, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}
	return rawCSVData
}

func parseDatasetCSV(rawCSVData [][]string, nFeatures, nClasses int) (*mat.Dense, *mat.Dense) {
	featureData := make([]float64, nFeatures*len(rawCSVData))
	labelData := make([]float64, nClasses*len(rawCSVData))

	var inputsIndex int
	var labelsIndex int

	for _, record := range rawCSVData {
		for i, val := range record {
			parsedVal, err := strconv.ParseFloat(val, 64)
			if err != nil {
				log.Fatal(err)
			}

			if i >= nFeatures {
				labelData[labelsIndex] = parsedVal
				labelsIndex++
				continue
			}

			featureData[inputsIndex] = parsedVal
			inputsIndex++
		}
	}

	return mat.NewDense(len(rawCSVData), nFeatures, featureData), mat.NewDense(len(rawCSVData), nClasses, labelData)
}

func loadTrainingDataset(fileName string, layout layout, split float64) (*mat.Dense, *mat.Dense, *mat.Dense, *mat.Dense) {
	rawCSVData := readDatasetCSV(fileName, layout.NClasses+layout.NFeatures)
	trainingRowLimit := int(float64(len(rawCSVData)) * (1 - split))

	traingData, trainingTarget := parseDatasetCSV(rawCSVData[1:trainingRowLimit], layout.NFeatures, layout.NClasses)
	testData, testTarget := parseDatasetCSV(rawCSVData[trainingRowLimit+1:], layout.NFeatures, layout.NClasses)

	return traingData, trainingTarget, testData, testTarget
}

func loadPredictionDataset(fileName string, layout layout) *mat.Dense {
	rawCSVData := readDatasetCSV(fileName, layout.NFeatures)

	data, _ := parseDatasetCSV(rawCSVData[1:], layout.NFeatures, layout.NClasses)

	return data
}
