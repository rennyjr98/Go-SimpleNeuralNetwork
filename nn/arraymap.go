package nn

import (
	"errors"
	"fmt"
	"math/rand"
)

type ArrayMap struct {
	Rows int
	Cols int
	Data [][]float64
}

func (arrMap *ArrayMap) InitByDimensions(rows int, cols int) {
	arrMap.Rows = rows
	arrMap.Cols = cols

	for i := 0; i < rows; i++ {
		row := make([]float64, cols)
		arrMap.Data = append(arrMap.Data, row)
	}
}

func (arrMap *ArrayMap) InitByData(data [][]float64) {
	arrMap.Rows = len(data)
	arrMap.Cols = len(data[0])
	arrMap.Data = data
}

func (arrMap *ArrayMap) Copy() [][]float64 {
	arrMapCopy := make([][]float64, arrMap.Rows, arrMap.Cols)
	for row, rowVal := range arrMap.Data {
		for col, colVal := range rowVal {
			arrMapCopy[row][col] = colVal
		}
	}
	return arrMapCopy
}

func (arrMap *ArrayMap) Transpose(x ArrayMap) ArrayMap {
	transposedArrMap := new(ArrayMap)
	transposedArrMap.InitByDimensions(x.Cols, x.Rows)
	for row, rowVal := range transposedArrMap.Data {
		for col := range rowVal {
			transposedArrMap.Data[row][col] = x.Data[col][row]
		}
	}
	return *transposedArrMap
}

func (arrMap *ArrayMap) FromArray(array []float64) ArrayMap {
	newSet := new(ArrayMap)
	newSet.InitByDimensions(len(array), 1)
	for row, rowVal := range newSet.Data {
		for col := range rowVal {
			newSet.Data[row][col] = array[row]
		}
	}
	return *newSet
}

func (arrMap *ArrayMap) ToArray() []float64 {
	array := make([]float64, arrMap.Rows*arrMap.Cols)
	arrIndex := 0
	for _, rowVal := range arrMap.Data {
		for _, colVal := range rowVal {
			array[arrIndex] = colVal
			arrIndex++
		}
	}
	return array
}

func (arrMap *ArrayMap) Randomize() {
	for row, rowVal := range arrMap.Data {
		for col := range rowVal {
			arrMap.Data[row][col] = rand.Float64()*2 - 1
		}
	}
}

func (arrMap *ArrayMap) Substract(a ArrayMap, b ArrayMap) (*ArrayMap, error) {
	if a.Rows != b.Rows || a.Cols != b.Cols {
		return nil, errors.New("ArrayMap a has not the same dimensions as ArrayMap b")
	}

	result := new(ArrayMap)
	result.InitByDimensions(a.Rows, a.Cols)
	for row, rowVal := range a.Data {
		for col := range rowVal {
			result.Data[row][col] = a.Data[row][col] - b.Data[row][col]
		}
	}

	return result, nil
}

func (arrMap *ArrayMap) Add(x ArrayMap) error {
	if x.Rows != arrMap.Rows || x.Cols != arrMap.Cols {
		return errors.New("ArrayMap x has not the same dimensions as target ArrayMap")
	}

	for row, rowVal := range x.Data {
		for col := range rowVal {
			arrMap.Data[row][col] += x.Data[row][col]
		}
	}
	return nil
}

func (arrMap *ArrayMap) Multiply(x ArrayMap) error {
	if arrMap.Rows != x.Rows || arrMap.Cols != x.Cols {
		return errors.New("ArrayMap target has not the same dimensions as ArrayMap x")
	}

	for row, rowVal := range x.Data {
		for col, colVal := range rowVal {
			arrMap.Data[row][col] *= colVal
		}
	}
	return nil
}

func (arrMap *ArrayMap) MultiplyArrMaps(a ArrayMap, b ArrayMap) (*ArrayMap, error) {
	if a.Cols != b.Rows {
		return nil, errors.New("ArrayMap a has not the same rows as ArrayMap b cols")
	}

	result := new(ArrayMap)
	result.InitByDimensions(a.Rows, b.Cols)

	for row := 0; row < a.Rows; row++ {
		for col := 0; col < b.Cols; col++ {
			sum := 0.0
			for aCol := 0; aCol < a.Cols; aCol++ {
				sum += a.Data[row][aCol] * b.Data[aCol][col]
			}
			result.Data[row][col] = sum
		}
	}

	return result, nil
}

func (arrMap *ArrayMap) MultiplyConst(x float64) {
	for row, rowVal := range arrMap.Data {
		for col := range rowVal {
			arrMap.Data[row][col] *= x
		}
	}
}

func (arrMap *ArrayMap) ActFunc(actFunc *SigmoidActivationFunc) {
	for row, rowVal := range arrMap.Data {
		for col, colVal := range rowVal {
			arrMap.Data[row][col] = actFunc.Funct(colVal)
		}
	}
}

func (arrMap *ArrayMap) ActDFunc(x ArrayMap, actFunc *SigmoidActivationFunc) *ArrayMap {
	for row, rowVal := range x.Data {
		for col, colVal := range rowVal {
			x.Data[row][col] = actFunc.DFunct(colVal)
		}
	}

	resultArrMap := new(ArrayMap)
	resultArrMap.InitByData(x.Data)
	return resultArrMap
}

func (arrMap *ArrayMap) Mutate(rate float64) {
	for row, rowVal := range arrMap.Data {
		for col := range rowVal {
			if rand.Float64() < rate {
				arrMap.Data[row][col] += rand.NormFloat64()
			}
		}
	}
}

func (arrMap *ArrayMap) Print() {
	fmt.Println("*****************************")
	fmt.Println("  Rows: ", arrMap.Rows)
	fmt.Println("  Cols: ", arrMap.Cols)
	fmt.Println("  Data: ")
	fmt.Print("    [")
	for row, rowVal := range arrMap.Data {
		if row == 0 {
			fmt.Print("[")
		} else {
			fmt.Print("    [")
		}
		for _, colVal := range rowVal {
			fmt.Print(colVal, " ")
		}
		if row == len(arrMap.Data)-1 {
			fmt.Print("]")
		} else {
			fmt.Println("]")
		}
	}
	fmt.Println("]")
	fmt.Println("*****************************")
}
