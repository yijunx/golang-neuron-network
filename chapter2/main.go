package main

import (
	"fmt"

	t "gorgonia.org/tensor"
)

type Layer struct {
	rawWeights     []float32
	numberOfNodes  int
	numberOfInputs int
	rawBiases      []float32
}

func forward(layers []Layer, rawInputs []float32) (t.Tensor, error) {
	inputs := t.New(t.WithShape(len(rawInputs), 1), t.WithBacking(rawInputs))
	outputRaw := make([]int, len(rawInputs))
	output, _ := t.Add(inputs, t.WithBacking(outputRaw))
	var err error
	for _, layer := range layers {
		weights := t.New(t.WithShape(layer.numberOfNodes, layer.numberOfInputs), t.WithBacking(layer.rawWeights))
		biases := t.New(t.WithShape(layer.numberOfNodes, 1), t.WithBacking(layer.rawBiases))
		beforeBiases, _ := t.Dot(weights, inputs)
		output, err = t.Add(beforeBiases, biases)
	}
	return output, err
}

func main() {

	l1 := Layer{
		rawWeights: []float32{
			0.2, 0.8, -0.5, 1,
			0.5, -0.91, 0.26, -0.5,
			-0.26, -0.27, 0.17, 0.87,
		},
		numberOfNodes:  3,
		numberOfInputs: 4,
		rawBiases: []float32{
			2,
			3,
			0.5,
		},
	}

	l2 := Layer{
		rawWeights: []float32{
			0.1, -0.14, 0.5,
			-0.5, 0.12, -0.33,
			-0.44, 0.73, -0.13,
		},
		numberOfNodes:  3,
		numberOfInputs: 3, // cos previos layer only 3 nodes
		rawBiases: []float32{
			-1,
			2,
			-0.5,
		},
	}

	// input
	rawInputs := []float32{
		1,
		2,
		3,
		2.5,
	}

	layers := []Layer{l1, l2}

	output, _ := forward(layers, rawInputs)
	fmt.Println(output)

}

// export ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.21

// func main() {

// 	n11 := Neuron{
// 		Weights: []float32{0.2, 0.8, -0.5, 1},
// 		Bias:    2,
// 	}
// 	n12 := Neuron{
// 		Weights: []float32{0.5, -0.91, 0.26, -0.5},
// 		Bias:    3,
// 	}
// 	n13 := Neuron{
// 		Weights: []float32{-0.26, -0.27, 0.17, 0.87},
// 		Bias:    0.5,
// 	}

// 	l1 := Layer{
// 		Neurons: []Neuron{n11, n12, n13},
// 	}

// 	inputs := []float32{1, 2, 3, 2.5}

// 	var output []float32
// 	for _, n := range l1.Neurons {
// 		var val float32 = 0
// 		for i := range n.Weights {
// 			val += n.Weights[i] * inputs[i]
// 		}
// 		output = append(output,
// 			n.Bias+val)
// 	}
// 	fmt.Println(output)
// }
