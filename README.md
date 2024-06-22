# Train Student GPT

## Description

Train a small Generative Pre-trained Transformer to generate student lecture commentary data from SIGHT ([Wang et. al., 2023](https://github.com/rosewang2008/sight/)).

## Getting Started

To train the model, you can run:
~~~
python run.py --mode train
~~~

To use the model, you can run:
~~~
python run.py --mode generate --prompt "### Lec 29 | MIT 18.01 Single Variable Calculus, Fall 2007" --max_new_tokens 300
~~~


## Results


### Examples from Training Data
> Processed from SIGHT data in comments.json.

```
### 4. Factorization into A = LU
Thank you for your leasons!

### Lec 2 | MIT 18.01 Single Variable Calculus, Fall 2007
I sure will pay it back hundredfold. Thanks!!!

### 2. Conditioning and Bayes' Rule
amazing explanations
```

### Training

The results of training for 27 minutes on an NVIDIA A100-80GB:

![Loss Curve](/example/loss_curve.png)

The generations include several comments and titles which appear realistic relative to the model size.


### Examples from Generated Data
> Prompt was "### Lec 29 | MIT 18.01 Single Variable Calculus, Fall 2007". The following are the best generated video titles and comments chosen from the [model's output](example/generated.md).

```
### Lec 1 | MIT 18.01 Single Variable Calculus, Fall 2007
He the best.

### Lec 24 | MIT 18.01 Single Variable Calculus, Fall 2007
Thanks

### 1. Introduction to Statistics
this course 😂😂
```

## Credits:

Andrej Karpathy for model code [https://www.youtube.com/watch?v=kCc8FmEb1nY](https://www.youtube.com/watch?v=kCc8FmEb1nY)

Wang et. al., 2023 for data: [https://github.com/rosewang2008/sight/](https://github.com/rosewang2008/sight/)