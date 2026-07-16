# "Introspection" with lens

[Interpreting GPT: the logit lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) and
(possibly) [others](https://arxiv.org/abs/2001.09309) had proposed the idea to decoding intermediate layers and proposed
to name the tool or method as "lens".

Lens can be attached to a model by using `--lens`. Types of lens that are supported:

* _Identity_: $len(l) = W_U \space {norm}(h_l)$

* _Linear_: $len(l) = W_U \space {norm}(M h_l)$, where $M$ is a matrix of shape $d_{model} \times d_{model}$ to _rotate_ $h_l$.

Where $W_U$ is the the unembedding matrix and $h_l$ is the activation output of layer $l$. Note that
${softmax}(\cdot)$ is not used in $len(l)$.

## Identity lens

Identity lens can be used on any model by adding command option: `--lens identity LAYERS`.
`LAYERS` selects the layers on which lens are attached. It uses the same format as [`--layer_spec`](./fun.md#layer-shuffling) and `all` is a shortcut for all layers.

## Linear lens

$M$ is needed. Luckily, a bunch of pre-fitted [Jacobian lens](https://transformer-circuits.pub/2026/workspace/index.html) for specific models are [released](https://huggingface.co/neuronpedia/jacobian-lens/tree/main) under MIT license.

1. Download a model and convert it for chatllm.cpp.
1. Download the pre-fitted Jacobian lens for this model.
1. Convert the Jacobian lens by `convert_j_lens.py`:

    ```sh
    python convert_j_lens.py -i /path/to/model/file -o quantized_name.bin --name ...
    ```

1. Use the lens by adding command option: `--lens linear LAYERS /path/to/quantized_name.bin`.

## Using binding

It's not very fruitful to use the command line interface with lens. API is recommended.