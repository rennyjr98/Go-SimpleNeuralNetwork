package main

import (
	"fmt"
	"neuralnetwork/nn"
)

func main() {
	neuraln := new(nn.NeuralNetwork)
	neuraln.InitByNodes(4, 3, 3)
	for i := 0; i < 1000; i++ {
		for _, rowVal := range nn.Dataset_setosa {
			err := neuraln.Train(rowVal, []float64{1, 0, 0})
			if err != nil {
				break
			}
		}
		for _, rowVal := range nn.Dataset_versicolor {
			err := neuraln.Train(rowVal, []float64{0, 1, 0})
			if err != nil {
				break
			}
		}
		for _, rowVal := range nn.Dataset_verginica {
			err := neuraln.Train(rowVal, []float64{0, 0, 1})
			if err != nil {
				break
			}
		}
	}

	fmt.Print(neuraln.Predict([]float64{6.5, 2.8, 4.6, 1.5}))
}
