# Coursera - Generative AI with Large Language Models

### Week1 : Generative AI use cases, project lifecycle, model pre-training and scaling laws



**Generative AI and LLM**

- Examples for GenAI (in order to mimic the human ability)
  - chatbot
  - generate images

* GenAI works by **finding stastical patterns from massive dataset of content**.

* LLM have been trained on trillons of words with large amounts of computer power, billions of params.

* Foundation / base models : GPT, BERT, FLAN-T5, PaLM, LLaMa, BLOOM

To interact LLM, you write computer code with formalized syntax to interact with libraries and APIs. 



**LLM use cases and tasks**

- summarize conversations
- traditional translation between two different languages
- translate natural language to machine code
- information retrieval => named entity recognition, a word classification.

Note : smaller models can be fine tuned to perform well on specific focused tasks.



**RNN vs Transformer**

Previous generations of language models used recurrent neural networks. **RNNs** while powerful for their time, **were limited by the amount of compute and memory needed to perform well at generative tasks**. To successfully predict the next word, models need to see more than just the previous few words. **Models needs to have an understanding of the whole sentence or even the whole document**. The problem here is that language is complex.

In 2017, after the publication of **Attention is All You Need** from Google and the University of Toronto, the transformer architecture had arrived. It can be scaled efficiently to use multi-core GPUs, it can parallel process input data, making use of much larger training datasets, and crucially, it's **able to learn to pay attention to the meaning of the words it's processing**.



**Transformer**

**Building LLM using the transformer architecture** dramatically improved the performance of natural language tasks over the earlier generation of RNNs, and led to an explosion in regenerative capability. The power of the transformer architecture lies in its ability to **learn the relevance and context of all of the words in a sentence**. 

So before passing texts into the model to process, you must first **tokenize the words**. What's important is that **once you've selected a tokenizer to train the model, you must use the same tokenizer when you generate text**. 

Now that your input is represented as numbers, you can pass it to the **embedding layer**. This layer is a **trainable vector embedding space**, a high-dimensional space where **each token is represented as a vector and occupies a unique location within that space**. Each token ID in the vocabulary is matched to a multi-dimensional vector, and the intuition is that these vectors learn to encode the meaning and context of individual tokens in the input sequence. 

Each word has been matched to a **token ID**, and each token is mapped into a **vector**. In the original transformer paper, the vector size was actually 512. As you add the token vectors into the base of the encoder or the decoder, you also add **positional encoding**. 

The model processes each of the input tokens in parallel. So **by adding the positional encoding, you preserve the information about the word order and don't lose the relevance of the position of the word in the sentence**. 

Once you've **summed the input tokens and the positional encodings**, you pass the resulting vectors to the **self-attention layer**. Here, **the model analyzes the relationships between the tokens in your input sequence**. As you saw earlier, **this allows the model to attend to different parts of the input sequence to better capture the contextual dependencies between the words**.

The self-attention weights that are learned during training and stored in these layers reflect the importance of each word in that input sequence to all other words in the sequence. But this does not happen just once, the transformer architecture actually has **multi-headed self-attention**. This means that **multiple sets of self-attention weights or heads are learned in parallel independently of each other**. **The number of attention heads** included in the attention layer varies from model to model, but numbers **in the range of 12-100 are common**. The intuition here is that **each self-attention head will learn a different aspect of language**. For example, one head may see the relationship between the people entities in our sentence. Whilst another head may focus on the activity of the sentence. Whilst yet another head may focus on some other properties such as if the words rhyme. It's important to note that you don't dictate ahead of time what aspects of language the attention heads will learn. The weights of each head are randomly initialized and given sufficient training data and time, each will learn different aspects of language. While some attention maps are easy to interpret, (like the examples discussed here), others may not be. Now that all of the attention weights have been applied to your input data, **the output is processed through a fully-connected feed-forward network**. The **output of this layer is a vector of logits proportional to the probability score for each and every token in the tokenizer dictionary**. You can then pass these logits to a final softmax layer, where they are normalized into a probability score for each word. This output includes a probability for every single word in the vocabulary, so there's likely to be thousands of scores here. One single token will have a score higher than the rest. This is the most likely predicted token.

**A translation task or a sequence-to-sequence task**, which incidentally was **the original objective of the transformer architecture designers**.



**Attention is all you need**

The paper proposes a neural network architecture that replaces traditional recurrent neural networks (RNNs) and convolutional neural networks (CNNs) with **an entirely attention-based mechanism**. 

The Transformer model uses self-attention to compute representations of input sequences, which allows it to capture long-term dependencies and parallelize computation effectively. The authors demonstrate that their model achieves state-of-the-art performance on several machine translation tasks and outperform previous models that rely on RNNs or CNNs.

The Transformer architecture consists of an encoder and a decoder, each of which is composed of several layers. **Each layer consists of two sub-layers: a multi-head self-attention mechanism and a feed-forward neural network**. The multi-head self-attention mechanism allows the model to attend to different parts of the input sequence, while the feed-forward network applies a point-wise fully connected layer to each position separately and identically. 

**Transformer model also uses residual connections and layer normalization to facilitate training and prevent overfitting**. In addition, the authors introduce a positional encoding scheme that encodes the position of each token in the input sequence, enabling the model to capture the order of the sequence without the need for recurrent or convolutional operations.



**What is prompt ?**

**The text that you pass to an LLM** is known as a prompt. The space or memory that is available to the prompt is called the **context window**, and this is typically large enough for a few thousand words, but differs from model to model.

The model then predicts the next words, and because your prompt contained a question, this model generates an answer.

The output of the model is called a **completion**, and the act of using the model to generate text is known as **inference**.



**Prompt and prompt engineering**

The work to develop and improve the prompt is known as **prompt engineering**. 

**In-context learning** provides examples inside the context window.

- Zero-shot inference : input data within the prompt.
- One-shot inference : the inclusion of a single example.
- Few-shot inference : the inclusion of multiple examples. 

You have a limit on the amount of in-context learning that you can pass into the model. 



**Elements of a prompt**

A prompt is composed with the following components : 

- Instructions
- Context
- Input data 
- Output indicator 

```
Classify the text into neutral, postive or negative [Instructions]
Text[context]: I think the food was okay [Input data].
Sentiment [Output indicator]
```



**Generative configuration**

In the task of next word prediction: 

- **Max new tokens** : number of tokens that the model will generate.

- **Greedy sampling** : the token with the highest probability is selected. 
- **Random(-weighted) sampling** : select a token using a random-weighted strategy across the probabilities of all tokens 
- **Top-k and top-p sampling (with random-weighted strategies)** : k means number of tokens in ranking, p means a cumulative probability  
- **Temperature** : level of randomness, the higher the tempature, the higher the randomness.



**Generative AI project lifecycle**

1. Scope (define the use case)
2. Select (choose an existing model or pretrain your own)
3. Adapt and align model (prompt engineering, fine-tuning and align with human feedback and evaluate)
4. Application integration (Optimize and deploy model for inference, augment model and build LLM-powered applications)



Giga byte > terabyte > petabyte 



**3 variances of transformer model**

- **Encoder only (AutoEncoding)** using masked langugage modeling : BERT, RoBERTa. 
  - Sentiment analysis, NER, word classification
- **Decoder only models (Autoregressive Model)** using causal language modeling, the context is unidirectional : GPT
  - Text generation (strong in zero-shot inference abilities)
- **Encoder-decoder model (Seq2seq Model)** make random sequences of input tokens: T5 / BART 
  - Translation, text summarization, Q&A



Thare are some computational challenges of training LLMs, such as GPU memory during training for string model weights, to reduce the memory required during training, we introduce a technique called **quantization** :

- reduce required memory to store and train models
- projects original 32-bit floating point numbers into lower precision spaces
- quantization-aware training (QAT) learns the quantization scaling during training
- **BFFLOAT16** is a popular choice

What else ? 

Use **multiple GPU compute strategies** when your model becomes too big to fit in a single GPU. 



**Pretraining your model from scratch will result in better models for highly specialized domains like law, medicine, finance or science.**

Example : BloombergGPT : domain adaptation in finance.

During the training of BloombergGPT, the authors used the Chinchilla Scaling Laws to guide the number of parameters in the model and the volume of training data, measured in tokens. The BloombergGPT project is a good illustration of pre-training a model for increased domain-specificity, and the challenges that may force **trade-offs against compute-optimal model and training configurations**. Chincilla papers carry out a detailed study the performance of language models of various sizes and quantities of training data.**The goal was to find the optimal number of parameters and volume of training data for a given compute budget**. 

The Chinchilla paper hints that **many of the 100 billion parameter large language models like GPT-3 may actually be over parameterized**, meaning they have more parameters than they need to achieve a good understanding of language and under trained so that they would benefit from seeing more training data. 

The authors hypothesized that **smaller models may be able to achieve the same performance as much larger ones if they are trained on larger datasets**. One important takeaway from the Chinchilla paper is that **the optimal training dataset size for a given model is about 20 times larger than the number of parameters in the model**. Chinchilla was determined to be compute optimal. For a 70 billion parameter model, the ideal training dataset contains 1.4 trillion tokens or 20 times the number of parameters. 

To scale our model, we need to jointly increase dataset size and model size, or they can become a bottleneck for each other.

When measuring compute budget, we can use **PetaFlops per second-Day** as a metric.



### Week 2 : instruction fine tuning

**What is the difference between pretraining and finetuning ?**

In contrast to pre-training, where you train the LLM using vast amounts of unstructured textual data via self-supervised learning, **fine-tuning is a supervised learning process** where you use a data set of labeled examples to update the weights of the LLM. The labeled examples are **prompt-completion pairs**, the fine-tuning process extends the training of the model to improve its ability to generate good completions for a specific task. 



**What is instruction fine-tuning ? **

**Instruction fine-tuning trains the model using examples that demonstrate how it should respond to a specific instruction.** Fine-tuning with instruction prompts is the most common way to fine-tune LLMs these days. From this point on, when you hear or see the term fine-tuning, you can assume that it always means instruction fine tuning.



**How does instruction fine-tuning work? **

1. Prepare and split data

The first step is to **prepare your training data**. There are many publicly available datasets that have been used to train earlier generations of language models, although most of them are not formatted as instructions. Luckily, **developers have assembled prompt template libraries that can be used to take existing datasets, for example, the large data set of Amazon product reviews and turn them into instruction prompt datasets for fine-tuning**. Prompt template libraries include many templates for different tasks and different datasets. Once you have your instruction data set ready, as with standard supervised learning, you divide the dataset into training validation and test splits. 

2. Model training, compare and improve results 

During fine tuning, you select prompts from your training dataset and pass them to the LLM, which then generates completions. 

Next, you compare the LLM completion with the response specified in the training data. So you can compare the distribution of the completion and that of the training label and **use the standard crossentropy function to calculate loss between the two token distributions**. And then use the calculated loss to update your model weights in standard backpropagation. **You'll do this for many batches of prompt completion pairs and over several epochs, update the weights so that the model's performance on the task improves**.

As in standard supervised learning, you can define separate evaluation steps to measure your LLM performance using the holdout validation dataset. This will give you the validation accuracy, and after you've completed your fine-tuning, you can perform a final performance evaluation using the holdout test data set .This will give you the test accuracy. 



**What is catastrophic forgetting ?**

Catastrophic forgetting is the full fine-tuning process modifies the weights of the original LLM, which **leads to great performance on the single fine-tuning task, it can degrade performance on other tasks**.



**How to avoid catastrophic forgetting ? **

- You might not have to.
- **Fine-tune on multiple tasks at the same time**.
- Consider PEFT (parameter efficient fine-tuning), which is a set of techniques that **preserves the weights of the original LLM and trains only a small number of task-specific adapter layers and parameters**. PEFT shows greater robustness to catastrophic forgetting since most of the pre-trained weights are left unchanged.



**What is FLAN ? **

FLAN, which stands for Fine-tuned LAnguage Net, is a specific set of instructions used to fine-tune different models. As an **instruction finetuning method**, it presents the results of its application. The study demonstrates that by fine-tuning the 540B PaLM model on 1836 tasks while incorporating Chain-of-Thought Reasoning data, FLAN achieves improvements in generalization, human usability, and zero-shot reasoning over the base model. 



**What are the evaluation metrics that we can use to mesure different task of LLM ? **

Use **ROUGE** for diagnostic evaluation of **summarization tasks** and **BLEU** for **translation tasks**. Other than that, there are some benchmarks that we can use to mesure and compare the performance of model.



**Benchmarks of LLM**

- **GLUE** : created to encourage the development of models that can generalize across multiple tasks, and you can use the benchmark to measure and compare the model performance. 

- **SuperGLUE** : consists of a series of tasks, some of which are not included in GLUE, and some of which are more challenging versions of the same tasks. SuperGLUE includes tasks such as multi-sentence reasoning, and reading comprehension. Both the GLUE and SuperGLUE benchmarks have leaderboards that can be used to compare and contrast evaluated models.

- **MMLU (Massive multitask language understanding)** : for modern LLMs, perform well models must possess extensive world knowledge and problem-solving ability. Models are tested on elementary mathematics, US history, computer science, law, and more. In other words, tasks that extend way beyond basic language understanding. 

- **BIGBench** : consists of 204 tasks, ranging through linguistics, childhood development, math, common sense reasoning, biology, physics, social bias, software development and more. BIG-bench comes in three different sizes, and part of the reason for this is to keep costs achievable, as running these large benchmarks can incur large inference costs.

- **HELM (Helistic evaluation of Language models)** : HELM takes a multimetric approach, measuring 7 metrics across 16 core scenarios, ensuring that trade-offs between models and metrics are clearly exposed. One important feature of HELM is that it assesses on metrics beyond basic accuracy measures, like precision of the F1 score. The benchmark also includes metrics for fairness, bias, and toxicity, which are becoming increasingly important to assess as LLMs become more capable of human-like language generation, and in turn of exhibiting potentially harmful behavior. HELM is a living benchmark that aims to continuously evolve with the addition of new scenarios, metrics, and models. 



**How does PEFT work ?**

With PEFT, most of the LLM weights are kept frozen. As a result, the number of trained parameters is much smaller than the number of parameters in the original LLM. In some cases, **just 15-20% of the original LLM weights**. This makes the **memory requirements for training much more manageable**. 

In fact, **PEFT can often be performed on a single GPU**. And because the original LLM is only slightly modified or left unchanged, PEFT is less prone to the catastrophic forgetting problems of full fine-tuning. 



**What are different categories of PEFT ?**

- **Selective** : select subset of initial LLM parameters to finetune
- **Reparametrization**: reparametrize model weights using a low-rank representation (LoRA)
- **Additive** : add trainable layers or params to model (Soft prompt)



**How does LoRA, a technique of PEFT in reparametrization category work ?**

LoRA stands for Low-Rank Adaptation, which uses **rank decomposition matrices to update the model parameters in an efficient way**. In Transformer achitecture, every parameter in self-attention layer is updated during fine-tuning. LoRA is a strategy that **reduces the number of parameters to be trained during fine-tuning by freezing all of the original model parameters and then injecting a pair of rank decomposition matrices alongside the original weights**. The dimensions of the smaller matrices are set so that **their product is a matrix with the same dimensions as the weights they're modifying**. **For inference, the two low-rank matrices are multiplied together to create a matrix with the same dimensions as the frozen weights.** You then add this to the original weights and replace them in the model with these updated values. 

This model has the same number of parameters as the original, so there is little to no impact on inference latency. Researchers have found that **applying LoRA to just the self-attention layers of the model is often enough to fine-tune for a task and achieve performance gains**. However, in principle, you can also use LoRA on other components like the feed-forward layers. But since most of the parameters of LLMs are in the attention layers, you get the biggest savings in trainable parameters by applying LoRA to these weights matrices. 



**How does rank decompostion matrices works in LoRA ? **

Let's look at a practical example using the transformer architecture described in the Attention is All You Need paper. The paper specifies that the transformer weights have dimensions of 512 by 64. This means that each weights matrix has 32,768 trainable parameters. 

If you use LoRA as a fine-tuning method with the rank equal to 8, you will instead train two small rank decomposition matrices whose small dimension is 8. This means that 

- Matrix A will have dimensions of 8 by 64, resulting in 512 total parameters. 

- Matrix B will have dimensions of 512 by 8, or 4,096 trainable parameters. 

By updating the weights of these new low-rank matrices instead of the original weights, you'll be training 4,608 parameters instead of 32,768 and 86% reduction. 



**What are benefits of LoRA ?**

Because LoRA allows you to significantly reduce the number of trainable parameters, you can often perform this method of parameter efficient fine tuning **with a single GPU and avoid the need for a distributed cluster of GPUs**. Since the rank-decomposition matrices are small, you can fine-tune a different set for each task and then switch them out at inference time by updating the weights. 

**Optimizing the choice of rank** is an ongoing area of research. The principles behind the method are useful not just for training LLMs, but for models in other domains.

You can also **combine LoRA with the quantization techniques to further reduce your memory footprint**. This is known as **QLoRA** in practice, PEFT is used heavily to minimize compute and memory resources. And ultimately reducing the cost of fine tuning, allowing you to make the most of your compute budget and speed up your development process.



**What is the limitation of prompt engineering ? **

Require a lot of **manual effort** to write and try different prompts, and the **length of context window is limited**. 



**What is prompt tuning ?**

**Add additional trainable tokens to prompts**, and leave it up to the supervised learning to determine the optimal values. **The set of trainable tokens is called a soft prompt, and it get pretended to embedding vectors that represents your input texts**. The soft prompt vectors have the same length as the embedding vectors of the language tokens. And including somewhere **between 20 and 100 virtual tokens** can be sufficient for good performance.  



**How does soft prompt, a technique of PEFT in additive category work ?**

Soft prompts are not fixed discrete words of natural language. Instead, you can think of them as **virtual tokens that can take on any value within the continuous multidimensional embedding space**. Through supervised learning, the model learns the values for these virtual tokens that maximize performance for a given task. 

In full fine tuning, the training dataset consists of input prompts and output completions or labels. The weights of the LLM are updated during supervised learning. In contrast with prompt tuning, the weights of the LLM are frozen and the underlying model does not get updated. Instead, the **embedding vectors of the soft prompt gets updated over time to optimize the model's completion of the prompt**. 

Soft prompts are very small on disk, so this kind of fine tuning is extremely efficient and flexible.

One potential issue to consider is the **interpretability of learned virtual tokens**. The trained tokens don't correspond to any known token, word, or phrase in the vocabulary of the LLM. However, an analysis of the **nearest neighbor tokens to the soft prompt location** shows that they form tight semantic clusters. In other words, **the words closest to the soft prompt tokens have similar meanings**. The words identified usually have some meaning related to the task, suggesting that the prompts are learning word like representations.



### Week3 :  Reinforcement learning and LLM-powered applications

**What is the adavantage of aligning models with human values ? **

Additional fine-tuning with human feedback helps to better align models with human preferences and to **increase the helpfulness, honesty, and harmlessness of the completions**. This further training can also help to **decrease the toxicity**, often models responses and **reduce the generation of incorrect information**. 



**How does RLHF work ? **

RLHF (Reinforcement Learning from Human Feedback) is a popular technique to finetune LLM with human feedback. In RLHF, **human labelers score a dataset of completions by the original model based on alignment criteria like helpfulness, harmlessness, and honesty**. This dataset is used to train the reward model that scores the model completions during the RLHF process.



**What is reinforcement learning ?**

Reinforcement learning is a type of machine learning in which **an agent learns to make decisions related to a specific goal by taking actions in an environment**, with the **objective of maximizing some notion of a cumulative reward**. The goal of reinforcement learning is for the agent to **learn the optimal policy for a given environment that maximizes their rewards**. In this framework, the agent continually learns from its experiences by taking actions, observing the resulting changes in the environment, and receiving rewards or penalties, based on the outcomes of its actions. By iterating through this process, the agent gradually refines its strategy or policy to make better decisions and increase its chances of success. 



**What is proximal policy optimization ?**

Proximal policy optimization (PPO for short) is popular algorithm for RLHF. **PPO optimizes a policy**, in this case the LLM, to be more aligned with human preferences. Over many iterations, PPO makes updates to the LLM. The updates are small and within a bounded region, **resulting in an updated LLM that is close to the previous version**, hence the name Proximal Policy Optimization. Keeping the changes within this small region result in a more stable learning. The goal is to update the policy so that the reward is maximized. 



**How to obtain human feedbacks in RLHF ?**

1. Define your model alignment criterion 

2. For the prompt-response set that you just generated, obtain human feedback through labeler workforce.



**What is KL-Divergence ?**

KL-Divergence, or Kullback-Leibler Divergence, is a concept often encountered in the field of reinforcement learning, particularly when using the Proximal Policy Optimization (PPO) algorithm. It is a **mathematical measure of the difference between two probability distributions, which helps us understand how one distribution differs from another**. In the context of PPO, KL-Divergence plays a crucial role in guiding the optimization process to ensure that the updated policy does not deviate too much from the original policy. 

To understand how KL-Divergence works, imagine we have two probability distributions: the distribution of the original LLM, and a new proposed distribution of an RL-updated LLM. **KL-Divergence measures the average amount of information gained when we use the original policy to encode samples from the new proposed policy**. By minimizing the KL-Divergence between the two distributions, PPO ensures that the updated policy stays close to the original policy, preventing drastic changes that may negatively impact the learning process.

A library that you can use to train transformer language models with reinforcement learning, using techniques such as PPO, is **TRL (Transformer Reinforcement Learning)**.



inference latency  = inference time



**Which techniques could we adopt for model optimizations  of LLM-powered applications before deployment ?**

[LLM optimization techniques before deploying it for inference](https://www.coursera.org/learn/generative-ai-with-llms/lecture/qojKp/model-optimizations-for-deployment)

- **Distillation** : have a larger teacher model to train a smaller student model, then use student model to lower the storage and compute budget.

  - The idea of distillation is to freeze the teacher model's weights and use it to generate completions for your training data. At the same time, generate completions for the training data using your student model. The knowledge distillation between teacher and student model is achieved by minimizing a loss function called the distillation loss. To calculate this loss, distillation uses the probability distribution over tokens that is produced by the teacher model's softmax layer. Now, the teacher model is already fine tuned on the training data. So the probability distribution likely closely matches the ground truth data and won't have much variation in tokens. That's why Distillation applies a little trick adding a temperature parameter to the softmax function as a higher temperature increases the creativity of the language the model generates. With a temperature parameter greater than one, the probability distribution becomes broader and less strongly peaked. This softer distribution provides you with a set of tokens that are similar to the ground truth tokens. In the context of Distillation, the teacher model's output is often referred to as soft labels and the student model's predictions as soft predictions. In parallel, you train the student model to generate the correct predictions based on your ground truth training data. Here, you don't vary the temperature setting and instead use the standard softmax function. Distillation refers to the student model outputs as the hard predictions and hard labels. The loss between these two is the student loss. The combined distillation and student losses are used to update the weights of the student model via back propagation. 

  - The key benefit of distillation methods is that the smaller student model can be used for inference in deployment instead of the teacher model. In practice, **distillation is not as effective for generative decoder models. It's typically more effective for encoder only models**, such as BERT that have a lot of representation redundancy. With distillation, you're training a second, smaller model to use during inference. You aren't reducing the model size of the initial LLM in any way.

- **Quantization (quantization aware training / QAT)**: transform a model's weight to a lower precision representation, such as 16-bit floating point or 8-bit integer, which can reduce the memory footprint of the model.
  
  - After training a model, you can perform PTQ (Post training quantization). PTQ transforms a model's weights to a lower precision representation, such as 16-bit floating point or 8-bit integer. Quantization can be applied to **just the model weights or to both weights and activation layers**.
  
- **Pruning** : remove redundant model parameters (weights) that contribute little to the model's performance. These are the weights with values very close to or equal to zero.

  

**Describe time and effort consumed in the lifecycle of GenAI Project**

- **Pretraining** : require high expertise and take from days to weeks to months (but at the most of times, you're prone to start with a foundation model)
- **Prompt engineering (assess the model's performance when starting with a foundation model)** : require less technical expertise, and no addtional training of the model.
- **Prompt tuning and fine-tuning (if the modeling isn't performing as you need)**: full finetuning or peft (LoRA or prompt tuning) depending on use cases, performance goals and compute budget. But since fine-tuning can be very successful with a relatively small training dataset, this phase could potentially be completed in a single day.
- **Reinforcement learning and human feedback** : require relevant expertise and similar time to fine-tuning if you can use an existing reward model. But it would take a long time to train a reward model from scratch because if need much efforts to gather human feedback. The goal is to update LLM model weights by adding a seperate reward model to align with human goals (helpful, honest and harmless).
- **Compression / optimization / deployment**



**What is the limitation of LLM and how could we reduce the negative effects ? **

**LLMs do not carry out mathematical operations**. They are still just trying to predict the next best token based on their training, and as a result, can easily get the answer wrong. 

**RAG (Retrieval Augmented Generation)** is a framework for building LLM powered systems that **make use of external data sources to overcome some of the limitations of these models** (like **knowledge cut-off issue**, or **model hallucinations** when it doesn't know the answer). A flexible and less expensive way to overcome knowledge cutoffs is to give your model access to additional external data at inference time. There are 2 considerations for using external data in RAG :

- Data must fit inside context window (split long sources into short chunks)

- Data must be in format that allows its relevance to be assessed at reference time : embedding vectors



**How does RAG work ?**

RAG methods take the small chunks of external data and process them through the LLM, to **create embedding vectors** for each. These new representations of the data can be stored in structures called **vector stores**, which allow for **fast searching of datasets and efficient identification of semantically related text**. Vector databases are a particular implementation of a vector store where each vector is also identified by a key. This can allow, for instance, the text generated by RAG to also include a citation for the document from which it was received. 



**What is PAL ?**

PAL (program-aided language models) is an interesting framework for **augmenting LLMs** as complex reasoning can be challenging for LLMs, especially for problems that involve multiple steps or mathematics. The strategy behind PAL is to have the LLM generate completions where **reasoning steps are accompanied by computer code**.

To prepare for inference with PAL, you'll format your prompt to contain one or more examples. Each example should contain a question followed by reasoning steps in lines of Python code that solve the problem. Next, you will append the new question that you'd like to answer to the prompt template.



In general, connecting LLMs to external applications allows the model to interact with the broader world, extending their utility beyond language tasks. LLMs can be used to trigger actions when given the ability to interact with APIs. LLMs can also connect to other programming resources. For example, a Python interpreter that can enable models to incorporate accurate calculations into their outputs. 



**What is chain of thought prompting ?**

Asking the model to mimic reasoning behavior is known as chain of thought prompting. It works by **including a series of intermediate reasoning steps** into any examples that you use for one or few-shot inference. 



**What are the benefits of  chain of thought prompting ?**

Chain of thought prompting is a powerful technique that **improves the ability of your model to reason through problems**. While this can greatly improve the performance of your model, the limited math skills of LLMs can still cause problems if your task requires accurate calculations, like totaling sales on an e-commerce site, calculating tax, or applying a discount. You can overcome this limitation by **allowing your model to interact with external applications** that are good at math, like a Python interpreter. 



**How does orchestrator work ?**

The orchestrator is a technical component that can **manage the flow of information and the initiation of calls to external data sources or applications**. It can also decide what actions to take based on the information contained in the output of the LLM.The LLM doesn't really have to decide to run the code, it just has to write the script which the orchestrator then passes to the external interpreter to run.



**What is ReAct and how does it work ?** 

ReAct is a novel approach that **integrates verbal reasoning and interactive decision making in LLMs**. While LLMs have excelled in language understanding and decision making, the combination of reasoning and acting has been neglected. **ReAct enables LLMs to generate reasoning traces and task-specific actions, leveraging the synergy between them**. The approach demonstrates superior performance over baselines in various tasks, **overcoming issues like hallucination and error propagation**. ReAct outperforms imitation and reinforcement learning methods in interactive decision making, even with minimal context examples. **It not only enhances performance but also improves interpretability, trustworthiness, and diagnosability by allowing humans to distinguish between internal knowledge and external information**.



**What are the building blocks for creating LLM-powered applications ?** 

You'll require several key components to create end-to-end solutions for your applications, starting with the **infrastructure layer**. This layer provides the **compute, storage, and network to serve up your LLMs, as well as to host your application components**. You can make use of your **on-premises infrastructure** for this or have it provided for you via **on-demand and pay-as-you-go Cloud services**. Next, you'll include the LLM you want to use in your application. These could include foundation models, as well as the models you have adapted to your specific task. 

The models are deployed on the appropriate infrastructure for your inference needs. Taking into account whether you need real-time or near-real-time interaction with the model. You may also have the need to **retrieve information from external sources**, such as RAG. Your application will return the completions from your LLM to the user or consuming application. Depending on your use case, **you may need to implement a mechanism to capture and store the outputs**. For example, you could build the capacity to store user completions during a session to augment the fixed contexts window size of your LLM. You can also gather feedback from users that may be useful for additional fine-tuning, alignment, or evaluation as your application matures. Next, you may need to use additional tools and frameworks for LLM that help you easily implement some of the techniques discussed in this course. As an example, you can use LangChain built-in libraries to implement techniques like ReAct or chain of thought prompting. You may also utilize model hubs which allow you to centrally manage and share models for use in applications. 

In the final layer, you typically have some type of **user interface** that the application will be consumed through, such as a website or a rest API. This layer is where you'll also include the **security components required for interacting with your application**. At a high level, this architecture stack represents the various components to consider as part of your generative AI applications. Your users, whether they are human end-users or other systems that access your application through its APIs, will interact with this entire stack. As you can see, the model is typically only one part of the story in building end-to-end generative AI applications.

Frameworks like LangChain are making it possible to quickly build, deploy, and test LLM powered applications, and it's a very exciting time for developers. 
