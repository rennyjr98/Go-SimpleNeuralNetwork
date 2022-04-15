package nn

type NeuralNetwork struct {
	InputNodes   int
	HiddenNodes  int
	OutputNodes  int
	learningRate float64

	WeightIH *ArrayMap
	WeightHO *ArrayMap
	BiasH    *ArrayMap
	BiasO    *ArrayMap

	sigmoid *SigmoidActivationFunc
}

func (nn *NeuralNetwork) InitByNodes(inputNodes int, hiddenNodes int, outputNodes int) {
	nn.InputNodes = inputNodes
	nn.HiddenNodes = hiddenNodes
	nn.OutputNodes = outputNodes

	nn.WeightIH = new(ArrayMap)
	nn.WeightIH.InitByDimensions(hiddenNodes, inputNodes)
	nn.WeightIH.Randomize()
	nn.WeightHO = new(ArrayMap)
	nn.WeightHO.InitByDimensions(outputNodes, hiddenNodes)
	nn.WeightHO.Randomize()

	nn.BiasH = new(ArrayMap)
	nn.BiasH.InitByDimensions(hiddenNodes, 1)
	nn.BiasH.Randomize()
	nn.BiasO = new(ArrayMap)
	nn.BiasO.InitByDimensions(outputNodes, 1)
	nn.BiasO.Randomize()
	nn.learningRate = 0.1
}

func (nn *NeuralNetwork) InitByNN(nnb *NeuralNetwork) {
	nn.InputNodes = nnb.InputNodes
	nn.HiddenNodes = nnb.HiddenNodes
	nn.OutputNodes = nnb.OutputNodes

	nn.WeightIH = nnb.WeightIH
	nn.WeightHO = nnb.WeightHO

	nn.BiasH = nnb.BiasH
	nn.BiasO = nnb.BiasO
	nn.learningRate = 0.1
}

func (nn *NeuralNetwork) Predict(input []float64) ([]float64, error) {
	inputArrMap := new(ArrayMap).FromArray(input)
	hiddenArrMap, err := new(ArrayMap).MultiplyArrMaps(*nn.WeightIH, inputArrMap)
	if err != nil {
		return nil, err
	}
	hiddenArrMap.Add(*nn.BiasH)
	hiddenArrMap.ActFunc(nn.sigmoid)

	outputs, err := new(ArrayMap).MultiplyArrMaps(*nn.WeightHO, *hiddenArrMap)
	if err != nil {
		return nil, err
	}
	outputs.Add(*nn.BiasO)
	outputs.ActFunc(nn.sigmoid)
	return outputs.ToArray(), nil
}

func (nn *NeuralNetwork) Train(input []float64, target []float64) error {
	inputArrMap := new(ArrayMap).FromArray(input)
	hiddenArrMap, err := new(ArrayMap).MultiplyArrMaps(*nn.WeightIH, inputArrMap)
	if err != nil {
		return err
	}
	hiddenArrMap.Add(*nn.BiasH)
	hiddenArrMap.ActFunc(nn.sigmoid)

	outputs, err := new(ArrayMap).MultiplyArrMaps(*nn.WeightHO, *hiddenArrMap)
	if err != nil {
		return err
	}
	outputs.Add(*nn.BiasO)
	outputs.ActFunc(nn.sigmoid)

	targetArrMap := new(ArrayMap).FromArray(target)
	// Calculate the error
	// ERROR = TARGET - OUTPUTs
	outputErrs, err := new(ArrayMap).Substract(targetArrMap, *outputs)
	if err != nil {
		return err
	}

	// Calculate gradient
	gradients := new(ArrayMap).ActDFunc(*outputs, nn.sigmoid)
	err = gradients.Multiply(*outputErrs)
	if err != nil {
		return err
	}
	gradients.MultiplyConst(nn.learningRate)

	// Calculate Deltas
	hiddenT := new(ArrayMap).Transpose(*hiddenArrMap)
	weightHODeltas, err := new(ArrayMap).MultiplyArrMaps(*gradients, hiddenT)
	if err != nil {
		return err
	}
	// Adjust the weight by deltas
	nn.WeightHO.Add(*weightHODeltas)
	// Adjust the bias by its deltas
	nn.BiasO.Add(*gradients)

	// Calculate the hidden layer errors
	whoT := new(ArrayMap).Transpose(*nn.WeightHO)
	hiddenErrs, err := new(ArrayMap).MultiplyArrMaps(whoT, *outputErrs)
	if err != nil {
		return err
	}

	// Calculate hidden gradients
	hiddenGradients := new(ArrayMap).ActDFunc(*hiddenArrMap, nn.sigmoid)
	hiddenGradients.Multiply(*hiddenErrs)
	hiddenGradients.MultiplyConst(nn.learningRate)

	// Calculate input hidden deltas
	inputT := new(ArrayMap).Transpose(inputArrMap)
	weightIHDeltas, err := new(ArrayMap).MultiplyArrMaps(*hiddenGradients, inputT)
	if err != nil {
		return err
	}
	nn.WeightIH.Add(*weightIHDeltas)
	nn.BiasH.Add(*hiddenGradients)
	return nil
}
