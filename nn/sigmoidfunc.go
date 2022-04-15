package nn

import "math"

type SigmoidActivationFunc struct{}

func (sigmoid *SigmoidActivationFunc) Funct(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func (sigmoid *SigmoidActivationFunc) DFunct(y float64) float64 {
	return y * (1 - y)
}
