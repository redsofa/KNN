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
Implements the k-Nearest Neighbors Machine Learning algorithm.
*/
package algo

import (
	"errors"
	"math"
	"sort"
)

/*
Source : Machine Learning in Action by Peter Harrington. Chapter 2.

KNN Algorithm :

For every point in our dataset:

	calculate the distance between inX and the current point
	sort the distances in increasing order
	take k items with lowest distances to inX
	find the majority class among these items
	return the majority class as our prediction for the class of inX
*/

/* KNN Type*/
type KNN struct {
	TrainingData []*TrainingData
	NearestCount int
}

func NewKNN(
	trainingData []*TrainingData,
	nearestCount int) (*KNN, error) {

	//Some basic input validation
	if nearestCount > len(trainingData) {
		return nil, errors.New("Nearest neighbor count cannot be bigger than training set size")
	}

	if nearestCount < 1 {
		return nil, errors.New("Nearest count cannot be less than 1.")
	}

	if len(trainingData) < 2 {
		return nil, errors.New("Length of training data cannot be less than 2.")
	}

	knn := &KNN{}
	knn.TrainingData = trainingData
	knn.NearestCount = nearestCount
	return knn, nil
}

func (knn *KNN) euclideanDistances(inputs []float64) (*[]distance, error) {
	var distances []distance
	var sum float64
	numberOfInputs := len(inputs)
	numberOfTrainingData := len(knn.TrainingData)

	for j := 0; j < numberOfTrainingData; j++ {
		sum = 0
		for i := 0; i < numberOfInputs; i++ {
			sum += math.Pow(inputs[i]-knn.TrainingData[j].Inputs[i], 2)
		}
		distances = append(distances, distance{math.Sqrt(sum), knn.TrainingData[j].Label}) // math.Sqrt(sum))
	}
	sort.Sort(DistanceSorter(distances))
	return &distances, nil
}

func (knn *KNN) Classify(inputs []float64) (*string, error) {
	if len(inputs) != len(knn.TrainingData[0].Inputs) {
		return nil, errors.New("Input data length has to be the same as training data length.")
	}

	//Note that the euclidean distances collection is sorted
	d, err := knn.euclideanDistances(inputs)
	if err != nil {
		return nil, err
	}

	var labelCounts map[string]int
	labelCounts = make(map[string]int)

	for i := 0; i < knn.NearestCount-1; i++ {
		label := (*d)[i].Label
		labelCounts[label] += 1
	}

	var maxCount int = 0
	var maxLabel string
	for k, v := range labelCounts {
		if v > maxCount {
			maxCount = v
			maxLabel = k
		}
	}

	var retVal = maxLabel
	return &retVal, nil
}

/* TrainingData Type */
type TrainingData struct {
	Inputs []float64
	Label  string
}

func NewTrainingData(inputs []float64, label string) *TrainingData {
	return &TrainingData{
		Inputs: inputs,
		Label:  label,
	}
}

/* Euclidean Distance Type */
type distance struct {
	Distance float64
	Label    string
}

/* Distance Sorter Type */
//Distance collection type which implements sort interface
type DistanceSorter []distance

func (d DistanceSorter) Len() int {
	return len(d)
}

func (d DistanceSorter) Less(i, j int) bool {
	return d[i].Distance < d[j].Distance
}

func (d DistanceSorter) Swap(i, j int) {
	d[i], d[j] = d[j], d[i]
}
