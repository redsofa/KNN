/*Copyright 2016 Rene Richard

This file is part of knn.

knn is free software: you can redistribute it and/or modify
it under the terms of the Apache License as published by the Apache Software
Foundation, either version 2.0 of the License, or (at your option) any later
version.

knn is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
the Apache License for more details.

You should have received a copy of the the Apache License
along with knn.
If not, please see <http://www.apache.org/licenses/>.
*/

/*
Implements the k-Nearest Neighbors Machine Learning algorithm call.
*/
package main

import (
	"fmt"
	algo "github.com/redsofa/knn/algo"
	version "github.com/redsofa/knn/version"
)

func generateTrainingData() []*algo.TrainingData {
	var trainingData []*algo.TrainingData
	var instance *algo.TrainingData

	instance = algo.NewTrainingData([]float64{0.0, 1.1}, "A")
	trainingData = append(trainingData, instance)

	instance = algo.NewTrainingData([]float64{1.0, 1.0}, "A")
	trainingData = append(trainingData, instance)

	instance = algo.NewTrainingData([]float64{0, 0}, "B")
	trainingData = append(trainingData, instance)

	instance = algo.NewTrainingData([]float64{0, 0.1}, "B")
	trainingData = append(trainingData, instance)

	return trainingData
}

func main() {
	println("A k-Nearest Neighbors algorithm implementation - Version :" + version.APP_VERSION)

	k := 2
	trainingData := generateTrainingData()

	var err error

	knn, err := algo.NewKNN(trainingData, k)
	if err != nil {
		fmt.Println(err)
	}

	var label *string
	label, err = knn.Classify([]float64{1, 1})
	println(fmt.Sprintf("The predicted label is %s for input of {1,1}", *label))

	label, err = knn.Classify([]float64{0, 0})
	println(fmt.Sprintf("The predicted label is %s for input of {0,0}", *label))

	label, err = knn.Classify([]float64{4, 6})
	println(fmt.Sprintf("The predicted label is %s for input of {4,6}", *label))
}
