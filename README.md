# Single-Layer Perceptron & the XOR Problem

<summary>Table of Contents</summary>

1. [What is XOR?](#what-is-xor)
2. [Linear Separability & Why XOR Fails](#linear-separability--why-xor-fails)
3. [Code Walkthrough](#code-walkthrough)
4. [Observed Results: 50% Accuracy](#observed-results-50-accuracy)
5. [Theoretical Best: 75% Accuracy](#theoretical-best-75-accuracy)

## What is XOR?

The **exclusive OR (XOR)** is a binary operation on two bits:

| Input A | Input B | A XOR B |
| :-----: | :-----: | :-----: |
|    0    |    0    |    0    |
|    0    |    1    |    1    |
|    1    |    0    |    1    |
|    1    |    1    |    0    |

- Outputs 1 **only** when exactly one input is 1.

## Linear Separability & Why XOR Fails

A dataset is **linearly separable** if you can draw a straight line (in 2D) that perfectly splits the “0”s from the “1”s.

- **XOR points** sit at the corners of a square:
  - (0,0) → 0
  - (0,1) → 1
  - (1,0) → 1
  - (1,1) → 0

Try to draw one line: whichever way you tilt, it will always misclassify at least one corner.

<img src="images/Screenshot 2025-05-21 at 12.39.26.png" width="300" alt="XOR not linearly separable"/>

## Code Walkthrough

<!-- <summary>Show code for the single-layer perceptron</summary> -->
This is XOR implementation using single-layer perceptron.

A single-layer perceptron computes a weighted sum of its inputs plus a bias, applies a step activation to produce a prediction, then updates its weights and bias by multiplying the prediction error with the inputs (and learning rate).

Let me explain the training using an example data point (x,y) = ([1,0],1)

- Initial weights: w1 = 0.2, w2 = -0.4, b = 0.1, since we are using random weights and bias
- Learning rate: η = 0.1

1. Weighted Sum:    
- z = w1.x1 + w2.x2 + b = ((0.2).1) + ((-0.4).0) + 0.1 = 0.3

2. Prediction:      
- ŷ = 1 if z >= 0 else 0

3. Error:           
- e = y - ŷ = 1 - 1 = 0

4. Update rule (only if e ≠ 0):     
- wj += η * e * xj 
- b  += η * e

Since e = 0, weights and bias remain unchanged.

Now for a misclassified sample (x, y) = ([0, 1], 1):

1. Weighted sum:    
- z = ((0.2).0) + ((-0.4).1) + 0.1 = -0.3

2. Prediction: 
- ŷ = 0

3. Error: 
- e = 1 - 0 = 1

4. Update:          
- w1 ← 0.2 + (0.1).(1).0  = 0.2
- w2 ← -0.4 + (0.1).(1).1 = -0.3
- b  ← 0.1 + (0.1).1   = 0.2

The boundary shifts so that [0, 1] will be correctly classified next pass.

When trained on the XOR dataset, a single-layer perceptron typically achieves around 50% accuracy due to its inability to model non-linear decision boundaries.

## Observed Results: 50% Accuracy

<img src="images/Screenshot 2025-05-21 at 12.45.48.png" width="500" alt="XOR not linearly separable"/>

- Result: Only 2 out of 4 points classified correctly (50%).
- Why 50%? Every straight line can slice off at most two “1”s from two “0”s in this arrangement.

Because no single linear boundary can classify all four XOR points correctly, each time the perceptron updates its weights to fix one misclassified example, it inevitably misclassifies another. This back-and-forth continues every epoch, causing the accuracy to oscillate rather than converge, since the perceptron learning rule only guarantees convergence when the data are linearly separable. This oscillation can be thought of as the model getting “stuck” between local improvements: adjusting weights to help one example inevitably hurts another. As a result, the perceptron never converges and simply fluctuates between suboptimal outcomes, with the final reported accuracy merely reflecting a snapshot of the last iteration.

## Theoretical Best: 75% Accuracy

Even if you pick the BEST possible line, you can at most get 3 points correct:
-Choose any three corners that lie roughly on one side of a line.
-The 4th corner will lie on the wrong side—no way to include all four.

Mathematically, for any weight vector w and bias b, the decision rule for XOR:

XOR = a(x_1 w_1 + x_2 w_2 + b)

where, 
- x_1, x_2 represents the input values
- w_1, w_2 represents the weights
- b represents the bias
- a represents the activation function

Since XOR labels are:

{(0,0): 0, (0,1): 1, (1,0): 1, (1,1): 0}

no linear inequality can satisfy all four. The best you can do is misclassify exactly one point, yielding

3/4 = 75% accuracy

<img src="images/Screenshot 2025-05-21 at 12.56.28.png" width="500" alt="XOR not linearly separable"/>
​

For example, the test point [0.1, 0.9] falls into the ‘1’ region of that line, so it’s also predicted as 1.

## Solving XOR with Single-Layer Perceptron Using PReLU

To overcome this, we implemented a custom single-layer perceptron that uses a non-linear activation function based on the absolute value of the weighted sum:

- This is a special case of the Parametric ReLU (PReLU) where a = -1, making the activation function behave like f(z) = |z|.
- This transforms the output space non-linearly, enabling the model to draw more flexible decision boundaries even with just one layer.

Keep in mind that this approach achieves 100% accuracy only when we fix the PReLU parameter a = -1, which effectively computes the absolute difference |x_1 - x_2|.

If instead we treat a as a learnable parameter, the model no longer consistently computes the absolute value. As a result, the transformed feature may not perfectly match the XOR behavior, and the accuracy starts to fluctuate again—similar to the original unmodified perceptron.

# Key Idea:

Add a new input feature:

x_3 = |x_1 - x_2|

| x_1     | x_2     | x_3 = abs(x_1 - x_2) | A XOR B |
| :-----: | :-----: |       :-----:     | :-----: |
|    0    |    0    |    0              |    0    |
|    0    |    1    |    1              |    1    |
|    1    |    0    |    1              |    1    |
|    1    |    1    |    0              |    0    |


This new feature perfectly matches the XOR output. Now the perceptron can simply learn a linear boundary in this setting such as:

predict 1 if x_3 ≥ 0.5

For model training, we are using weights: [1.0,-1.0] and bias: 0, an under these parameters we can acheive 100% accuracy.   

<img src="/Users/rohanxc/Developer/Semi-technical-Interview-Task/images/Screenshot 2025-05-21 at 21.58.17.png" width="500" alt="XOR not linearly separable"/>

# Demonstration of why this feature mapping works

Let me explain the training using an example data point (x,y) = ([1,0],1)

We have x_1 = 1, x_2 = 0, and y = 1

1. Compute weighted sum:

- weighted sum = x_1.w_1 + x_2.w_2 + b = 1.1 + 0.(-1) + b = 1

2. Modified PReLU activation function apply:

Since PReLU parameter a = -1, activation function will return abs(x_1 - x_2)

- predicted_output = abs(weighted sum) = abs(1) = 1

As you can see the prediction came correct and just like this other input will also give correct prediction given we use a = -1.


## Conclusion

By modifying the activation function alone, we allow a single-layer perceptron to "bend" the decision boundary in a way that correctly classifies all XOR cases. This demonstrates how critical activation functions are in neural networks—not just depth.

## Citation

```
@article{pinto2024prelu,
  title   = {PReLU: Yet Another Single-Layer Solution to the XOR Problem},
  author  = {Pinto, Rafael C. and Tavares, Anderson R.},
  journal = {arXiv preprint},
  volume  = {arXiv:2409.10821},
  year    = {2024},
  month   = {sep},
  url     = {https://doi.org/10.48550/arXiv.2409.10821}
}
```