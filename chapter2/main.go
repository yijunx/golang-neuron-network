package main

import "fmt"

type Neuron struct {
	Weights []float32
	Bias    float32
}

type Layer struct {
	Neurons []Neuron
}

func main() {

	n11 := Neuron{
		Weights: []float32{0.2, 0.8, -0.5, 1},
		Bias:    2,
	}
	n12 := Neuron{
		Weights: []float32{0.5, -0.91, 0.26, -0.5},
		Bias:    3,
	}
	n13 := Neuron{
		Weights: []float32{-0.26, -0.27, 0.17, 0.87},
		Bias:    0.5,
	}

	l1 := Layer{
		Neurons: []Neuron{n11, n12, n13},
	}

	inputs := []float32{1, 2, 3, 2.5}

	var output []float32
	for _, n := range l1.Neurons {
		var val float32 = 0
		for i := range n.Weights {
			val += n.Weights[i] * inputs[i]
		}
		output = append(output,
			n.Bias+val)
	}
	fmt.Println(output)
}
