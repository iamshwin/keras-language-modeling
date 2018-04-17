# keras-language-modeling

Some code for doing language modeling with Keras, in particular for question-answering tasks. Forked from the blog/code [here](https://codekansas.github.io/blog/2016/language.html).

### Getting started 

```
pip install -r requirements.txt
# Clone InsuranceQA dataset
git clone https://github.com/codekansas/insurance_qa_python
export INSURANCE_QA=$(pwd)/insurance_qa_python

python insurance_qa_eval.py --config config.embedding_model
```



### Stuff that might be of interest

 - `attention_lstm.py`: Attentional LSTM, based on one of the papers referenced in the blog post and others. One application used it for [image captioning](http://arxiv.org/pdf/1502.03044.pdf). It is initialized with an attention vector which provides the attention component for the neural network.
 - `insurance_qa_eval.py`: Evaluation framework for the InsuranceQA dataset. To get this working, clone the [data repository](https://github.com/codekansas/insurance_qa_python) and set the `INSURANCE_QA` environment variable to the cloned repository. Changing `config` will adjust how the model is trained.
 - `keras-language-model.py`: The `LanguageModel` class uses the `config` settings to generate a training model and a testing model. The model can be trained by passing a question vector, a ground truth answer vector, and a bad answer vector to `fit`. Then `predict` calculates the similarity between a question and answer. Override the `build` method with whatever language model you want to get a trainable model. Examples are provided at the bottom, including the `EmbeddingModel`, `ConvolutionModel`, and `RecurrentModel`.

### Additionally

 - The official implementation can be found [here](https://github.com/white127/insuranceQA-cnn-lstm)

### Data

 - L6 from [Yahoo Webscope](http://webscope.sandbox.yahoo.com/)
 - [InsuranceQA data](https://github.com/shuzi/insuranceQA)
   - [Pythonic version](https://github.com/codekansas/insurance_qa_python)

