<div align="center">
  <!-- <h1><b> Time-LLM </b></h1> -->
  <!-- <h2><b> Time-LLM </b></h2> -->
  <h2><b> Propulsion: Steering LLM with Tiny Fine-Tuning </b></h2>
</div>


<div align="center">

**[<a href="https://arxiv.org/abs/2409.10927">Paper Page</a>]**
**[<a href="https://github.com/Kowsher/Propulsion">Code</a>]**


</div>

<p align="center">

<img src="./figures/logo.png" width="70">

</p>

---
>
> 🙋 Please let us know if you find out a mistake or have any suggestions!
> 
> 🌟 If you find this resource helpful, please consider to star this repository and cite our research:

```
@article{kowsher2024propulsion,
  title={Propulsion: Steering LLM with Tiny Fine-Tuning},
  author={Kowsher, Md and Prottasha, Nusrat Jahan and Bhat, Prakash},
  journal={arXiv preprint arXiv:2409.10927},
  year={2024}
}
```

## Introduction
 Propulsion, a  parameter-efficient fine-tuning (PEFT) method designed to optimize task-specific performance while drastically reducing computational overhead


## Requirements
Use python 3.11 from MiniConda

- torch==2.3.0
- accelerate==0.33.0
- einops==0.7.0
- matplotlib==3.7.0
- numpy==1.23.5
- pandas==1.5.3
- scikit_learn==1.2.2
- scipy==1.12.0
- tqdm==4.65.0
- peft==0.12.0
- transformers==4.44.0
- deepspeed==0.15.1
- sentencepiece==0.2.0


To install all dependencies:
```
pip install -r requirements.txt
```

## Datasets
You can access the datasets from hugginface

## Quick Demos

To get started with `propulsion`, follow these simple steps:

1. **Import the necessary modules:**

    ```python
    import propulsion
    from transformers import RobertaForSequenceClassification
    ```

2. **Load a pre-trained model and apply PEFT:**

    ```python
    model = RobertaForSequenceClassification.from_pretrained('model_name')
    propulsion.PEFT(model)
    ```

3. **Now you're ready to fine-tune your model using `propulsion`.**

