# Research Paper Summaries

## Table of Contents

- [Artificial Intelligence and Machine Learning Techniques for Image Synthesis](#artificial-intelligence-and-machine-learning-techniques-for-image-synthesis)
  - [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](#deep-unsupervised-learning-using-nonequilibrium-thermodynamics)
  - [Learning Transferable Visual Models From Natural Language Supervision](#learning-transferable-visual-models-from-natural-language-supervision)
  - [Elucidating the Design Space of Diffusion-Based Generative Models](#elucidating-the-design-space-of-diffusion-based-generative-models)
  - [Learning to Generate and Transfer Data with Rectified Flow](#learning-to-generate-and-transfer-data-with-rectified-flow)
  - [Building Normalizing Flows With Stochastic Interpolants](#building-normalizing-flows-with-stochastic-interpolants)
  - [Scalable Diffusion Models with Transformers](#scalable-diffusion-models-with-transformers)
  - [Flow Matching in Latent Space](#flow-matching-in-latent-space)
  - [Tutorial on Diffusion Models for Imaging and Vision](#tutorial-on-diffusion-models-for-imaging-and-vision)
  - [Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction](#visual-autoregressive-modeling-scalable-image-generation-via-next-scale-prediction)
  - [Inference-Time Scaling for Diffusion Models beyond Scaling Denoising Steps](#inference-time-scaling-for-diffusion-models-beyond-scaling-denoising-steps)
  - [Flow Matching for Generative Modeling](#flow-matching-for-generative-modeling)
- [Quantization Techniques in AI/ML Research Papers](#quantization-techniques-in-ai/ml-research-papers)
  - [DeepShift: Towards Multiplication-Less Neural Networks](#deepshift-towards-multiplication-less-neural-networks)
  - [AdderNet: Do We Really Need Multiplications in Deep Learning?](#addernet-do-we-really-need-multiplications-in-deep-learning)
  - [VS-Q UANT: PER-VECTOR SCALED QUANTIZATION FOR ACCURATE LOW-PRECISION NEURAL NETWORK INFERENCE](#vs-q-uant-per-vector-scaled-quantization-for-accurate-low-precision-neural-network-inference)
  - [LLM.int8() : 8-bit Matrix Multiplication for Transformers at Scale](#llmint8--8-bit-matrix-multiplication-for-transformers-at-scale)
  - [GPTQ: A Curate Post-Training Quantization for Generative Pre-Trained Transformers](#gptq-a-curate-post-training-quantization-for-generative-pre-trained-transformers)
  - [Multiplication-Free Transformer Training via Piecewise Affine Operations](#multiplication-free-transformer-training-via-piecewise-affine-operations)
  - [A WQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](#a-wq-activation-aware-weight-quantization-for-llm-compression-and-acceleration)
  - [BitNet: Scaling 1-bit Transformers for Large Language Models](#bitnet-scaling-1-bit-transformers-for-large-language-models)
  - [FP6-LLM: Efficiently Serving Large Language Models Through FP6-Centric Algorithm-System Co-Design](#fp6-llm-efficiently-serving-large-language-models-through-fp6-centric-algorithm-system-co-design)
  - [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](#the-era-of-1-bit-llms-all-large-language-models-are-in-158-bits)
  - [SpinQuant: LLM Quantization with Learned Rotations](#spinquant-llm-quantization-with-learned-rotations)
  - [Scalable MatMul-free Language Modeling](#scalable-matmul-free-language-modeling)
  - [ShiftAddLLM: Accelerating Pretrained LLMs via Post-Training Multiplication-Less Reparameterization](#shiftaddllm-accelerating-pretrained-llms-via-post-training-multiplication-less-reparameterization)
  - [QTIP: Quantization with Trellises and Incoherence](#qtip-quantization-with-trellises-and-incoherence)
  - [VPTQ: Extreme Low-bit Vector Post-Training Quantization for Large Language Models](#vptq-extreme-low-bit-vector-post-training-quantization-for-large-language-models)
  - [How Numerical Precision Affects Mathematical Reasoning Capabilities of LLMs](#how-numerical-precision-affects-mathematical-reasoning-capabilities-of-llms)
  - [Low-Bit Quantization Favors Undertrained LLMs: Scaling Laws for Quantized LLMs with 100T Training Tokens](#low-bit-quantization-favors-undertrained-llms-scaling-laws-for-quantized-llms-with-100t-training-tokens)
  - [Least Squares Quantization in PCM](#least-squares-quantization-in-pcm)
  - [Stationary and Trellis Encoding for IID Sources and Simulation](#stationary-and-trellis-encoding-for-iid-sources-and-simulation)
  - [Trellis Coded Quantization of Memoryless and Gauss-Markov Sources](#trellis-coded-quantization-of-memoryless-and-gauss-markov-sources)
  - [ShiftAddNet: A Hardware-Inspired Deep Network](#shiftaddnet-a-hardware-inspired-deep-network)
  - [OCP Microscaling Formats (MX) Specification](#ocp-microscaling-formats-mx-specification)
- [Unsorted Papers](#unsorted-papers)
  - [Distilling the Knowledge in a Neural Network](#distilling-the-knowledge-in-a-neural-network)
  - [Using the Output Embedding to Improve Language Models](#using-the-output-embedding-to-improve-language-models)
  - [Trained Ternary Quantization](#trained-ternary-quantization)
  - [Proximal Policy Optimization Algorithms](#proximal-policy-optimization-algorithms)
  - [Averaging Weights Leads to Wider Optima and Better Generalization](#averaging-weights-leads-to-wider-optima-and-better-generalization)
  - [World Models](#world-models)
  - [SqueezeNext: Hardware-Aware Neural Network Design](#squeezenext-hardware-aware-neural-network-design)
  - [Deep k-Means: Re-Training and Parameter Sharing with Harder Cluster Assignments for Compressing Deep Convolutions](#deep-k-means-re-training-and-parameter-sharing-with-harder-cluster-assignments-for-compressing-deep-convolutions)
  - [Universal Transformers](#universal-transformers)
  - [Improving Neural Network Quantization without Retraining using Outlier Channel Splitting](#improving-neural-network-quantization-without-retraining-using-outlier-channel-splitting)
  - [Learned Step Size Quantization](#learned-step-size-quantization)
  - [All You Need Is a Few Shifts: Designing Efficient Convolutional Neural Networks for Image Classification](#all-you-need-is-a-few-shifts-designing-efficient-convolutional-neural-networks-for-image-classification)
  - [Character Region Awareness for Text Detection](#character-region-awareness-for-text-detection)
  - [Searching for MobileNetV3](#searching-for-mobilenetv3)
  - [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](#efficientnet-rethinking-model-scaling-for-convolutional-neural-networks)
  - [Large Memory Layers with Product Keys](#large-memory-layers-with-product-keys)
  - [ADDITIVE POWERS-OF-TWO QUANTIZATION: AN EFFICIENT NON-UNIFORM DISCRETIZATION FOR NEURAL NETWORKS](#additive-powers-of-two-quantization-an-efficient-non-uniform-discretization-for-neural-networks)
  - [Scaling Laws for Neural Language Models](#scaling-laws-for-neural-language-models)
  - [On Layer Normalization in the Transformer Architecture](#on-layer-normalization-in-the-transformer-architecture)
  - [GLU Variants Improve Transformer Feed Forward Networks](#glu-variants-improve-transformer-feed-forward-networks)
  - [A Neural Scaling Law from the Dimension of the Data Manifold](#a-neural-scaling-law-from-the-dimension-of-the-data-manifold)
  - [Language Models are Few-Shot Learners](#language-models-are-few-shot-learners)
  - [AdderSR: Towards Energy Efficient Image Super-Resolution](#addersr-towards-energy-efficient-image-super-resolution)
  - [ANIMAGE IS WORTH 16X16 W ORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE](#animage-is-worth-16x16-w-ords-transformers-for-image-recognition-at-scale)
  - [Scaling Laws for Autoregressive Generative Modeling](#scaling-laws-for-autoregressive-generative-modeling)
  - [Sparsity in Deep Learning: Pruning and growth for efficient inference and training in neural networks](#sparsity-in-deep-learning-pruning-and-growth-for-efficient-inference-and-training-in-neural-networks)
  - [Explaining Neural Scaling Laws](#explaining-neural-scaling-laws)
  - [Network Quantization with Element-wise Gradient Scaling](#network-quantization-with-element-wise-gradient-scaling)
  - [LeViT: a Vision Transformer in ConvNet's Clothing for Faster Inference](#levit-a-vision-transformer-in-convnets-clothing-for-faster-inference)
  - [ROFORMER: Enhanced Transformer with Rotary Position Embedding](#roformer-enhanced-transformer-with-rotary-position-embedding)
  - [DKM: Differentiable k-Means Clustering Layer for Neural Network Compression](#dkm-differentiable-k-means-clustering-layer-for-neural-network-compression)
  - [Primer: Searching for Efficient Transformers for Language Modeling](#primer-searching-for-efficient-transformers-for-language-modeling)
  - [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](#chain-of-thought-prompting-elicits-reasoning-in-large-language-models)
  - [Gradients Without Backpropagation](#gradients-without-backpropagation)
  - [Rare Gems: Finding Lottery Tickets at Initialization](#rare-gems-finding-lottery-tickets-at-initialization)
  - [Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer](#tensor-programs-v-tuning-large-neural-networks-via-zero-shot-hyperparameter-transfer)
  - [STaR: Self-Taught Reasoner Bootstrapping Reasoning With Reasoning](#star-self-taught-reasoner-bootstrapping-reasoning-with-reasoning)
  - [Training Compute-Optimal Large Language Models](#training-compute-optimal-large-language-models)
  - [Optimal Clipping and Magnitude-aware Differentiation for Improved Quantization-aware Training](#optimal-clipping-and-magnitude-aware-differentiation-for-improved-quantization-aware-training)
  - [Symbolic Discovery of Optimization Algorithms](#symbolic-discovery-of-optimization-algorithms)
  - [TinyStories: How Small Can Language Models Be and Still Speak Coherent English?](#tinystories-how-small-can-language-models-be-and-still-speak-coherent-english)
  - [Let's Verify Step by Step](#lets-verify-step-by-step)
  - [MiniLLM: Knowledge Distillation of Large Language Models](#minillm-knowledge-distillation-of-large-language-models)
  - [Patch n’ Pack: NaViT, a Vision Transformer for any Aspect Ratio and Resolution](#patch-n-pack-navit-a-vision-transformer-for-any-aspect-ratio-and-resolution)
  - [PyPIM: Integrating Digital Processing-in-Memory from Microarchitectural Design to Python Tensors](#pypim-integrating-digital-processing-in-memory-from-microarchitectural-design-to-python-tensors)
  - [Explaining Grokking through Circuit Efficiency](#explaining-grokking-through-circuit-efficiency)
  - [Training Chain-of-Thought via Latent-Variable Inference](#training-chain-of-thought-via-latent-variable-inference)
  - [REFT: Reasoning with Reinforced Fine-Tuning](#reft-reasoning-with-reinforced-fine-tuning)
  - [No Free Prune: Information-Theoretic Barriers to Pruning at Initialization](#no-free-prune-information-theoretic-barriers-to-pruning-at-initialization)
  - [ReLU2Wins: Discovering Efficient Activation Functions for Sparse LLMs](#relu2wins-discovering-efficient-activation-functions-for-sparse-llms)
  - [DISTILLM: Towards Streamlined Distillation for Large Language Models](#distillm-towards-streamlined-distillation-for-large-language-models)
  - [Grandmaster-Level Chess Without Search](#grandmaster-level-chess-without-search)
  - [Training Large Language Models for Reasoning through Reverse Curriculum Reinforcement Learning](#training-large-language-models-for-reasoning-through-reverse-curriculum-reinforcement-learning)
  - [Training Large Language Models for Reasoning through Reverse Curriculum Reinforcement Learning](#training-large-language-models-for-reasoning-through-reverse-curriculum-reinforcement-learning)
  - [Chain-of-Thought Reasoning without Prompting](#chain-of-thought-reasoning-without-prompting)
  - [Chain of Thought Empowers Transformers to Solve Inherently Serial Problems](#chain-of-thought-empowers-transformers-to-solve-inherently-serial-problems)
  - [MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases](#mobilellm-optimizing-sub-billion-parameter-language-models-for-on-device-use-cases)
  - [Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking](#quiet-star-language-models-can-teach-themselves-to-think-before-speaking)
  - [Reinforcement Learning from Reflective Feedback (RLRF): Aligning and Improving LLMs via Fine-Grained Self-Reflection](#reinforcement-learning-from-reflective-feedback-rlrf-aligning-and-improving-llms-via-fine-grained-self-reflection)
  - [The Unreasonable Ineffectiveness of the Deeper Layers](#the-unreasonable-ineffectiveness-of-the-deeper-layers)
  - [QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs](#quarot-outlier-free-4-bit-inference-in-rotated-llms)
  - [Mixture-of-Depths: Dynamically Allocating Compute in Transformer-Based Language Models](#mixture-of-depths-dynamically-allocating-compute-in-transformer-based-language-models)
  - [Talaria: Interactively Optimizing Machine Learning Models for Efficient Inference](#talaria-interactively-optimizing-machine-learning-models-for-efficient-inference)
  - [Training LLMs over Neurally Compressed Text](#training-llms-over-neurally-compressed-text)
  - [Physics of Language Models: Part 3.3 Knowledge Capacity Scaling Laws](#physics-of-language-models-part-33-knowledge-capacity-scaling-laws)
  - [Grokked Transformers are Implicit Reasoners: A Mechanistic Journey to the Edge of Generalization](#grokked-transformers-are-implicit-reasoners-a-mechanistic-journey-to-the-edge-of-generalization)
  - [MoEUT: Mixture-of-Experts Universal Transformers](#moeut-mixture-of-experts-universal-transformers)
  - [Accessing GPT-4 Level Mathematical Olympiad Solutions via Monte Carlo Tree Self-refine with LLaMa-3 8B: A Technical Report](#accessing-gpt-4-level-mathematical-olympiad-solutions-via-monte-carlo-tree-self-refine-with-llama-3-8b-a-technical-report)
  - [What Matters in Transformers? Not All Attention Is Needed](#what-matters-in-transformers-not-all-attention-is-needed)
  - [What Matters in Transformers?](#what-matters-in-transformers)
  - [ARES: Alternating Reinforcement Learning and Supervised Fine-Tuning for Enhanced Multi-Modal Chain-of-Thought Reasoning Through Diverse AI Feedback](#ares-alternating-reinforcement-learning-and-supervised-fine-tuning-for-enhanced-multi-modal-chain-of-thought-reasoning-through-diverse-ai-feedback)
  - [Q-Sparse: All Large Language Models can be Fully Sparsely-Activated](#q-sparse-all-large-language-models-can-be-fully-sparsely-activated)
  - [Large Language Monkeys: Scaling Inference Compute with Repeated Sampling](#large-language-monkeys-scaling-inference-compute-with-repeated-sampling)
  - [Scaling LLM Test-Time Compute Optimally Can Be More Effective than Scaling Model Parameters](#scaling-llm-test-time-compute-optimally-can-be-more-effective-than-scaling-model-parameters)
  - [Iteration of Thought: Leveraging Inner Dialogue for Autonomous Large Language Model Reasoning](#iteration-of-thought-leveraging-inner-dialogue-for-autonomous-large-language-model-reasoning)
  - [Training Language Models to Self-Correct via Reinforcement Learning](#training-language-models-to-self-correct-via-reinforcement-learning)
  - [GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models](#gsm-symbolic-understanding-the-limitations-of-mathematical-reasoning-in-large-language-models)
  - [Value Residual Learning for Alleviating Attention Concentration in Transformers](#value-residual-learning-for-alleviating-attention-concentration-in-transformers)
  - [Scaling Laws for Precision](#scaling-laws-for-precision)
  - [Convolutional Differentiable Logic Gate Networks](#convolutional-differentiable-logic-gate-networks)
  - [BitNet a4.8: 4-bit Activations for 1-bit LLMs](#bitnet-a48-4-bit-activations-for-1-bit-llms)
  - [A Hybrid-head Architecture for Small Language Models](#a-hybrid-head-architecture-for-small-language-models)
  - [Flow Matching Guide and Code](#flow-matching-guide-and-code)
  - [Training Large Language Models to Reason in a Continuous Latent Space](#training-large-language-models-to-reason-in-a-continuous-latent-space)
  - [µNAS: Constrained Neural Architecture Search for Microcontrollers](#nas-constrained-neural-architecture-search-for-microcontrollers)
  - [DeepSeek-V3 Technical Report](#deepseek-v3-technical-report)
  - [Memory Layers at Scale](#memory-layers-at-scale)
  - [Optimal Brain Damage](#optimal-brain-damage)

---

## Artificial Intelligence and Machine Learning Techniques for Image Synthesis

### Deep Unsupervised Learning using Nonequilibrium Thermodynamics

*Jascha Sohl-Dickstein, Eric A. Weiss, Niru Maheswaranathan, Surya Ganguli*

**Summary:** A central problem in machine learning involves modeling complex data-sets using highly flexible families of probability distributions in which learning, sampling, inference, and evaluation are still analytically or computationally tractable.

**ArXiv:** [1503.03585](https://arxiv.org/abs/1503.03585), [Local link](Papers%5Cimage_synthesis%5C1503.03585v8_Deep%20Unsupervised%20Learning%20using%20Nonequilibrium%20Thermodynamics.pdf), Hash: b1390adda57bbb669fa958add52328c3, *Added: 2025-01-17* 

### Learning Transferable Visual Models From Natural Language Supervision

*Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever*

**Summary:** State-of-the-art computer vision systems are trained to predict a fixed set of predetermined object categories. This restricted form of supervision limits their generality and usability since additional labeled data is needed to specify any other visual concept. Learning directly from raw text about images is a promising alternative which leverages a much broader source of supervision.

**ArXiv:** [2103.00020](https://arxiv.org/abs/2103.00020), [Local link](Papers%5Cimage_synthesis%5C2103.00020v1_CLIP_Learning%20Transferable%20Visual%20Models%20From%20Natural%20Language%20Supervision.pdf), Hash: 1fc956711b1d99f17c09102e6f73ff2e, *Added: 2025-01-17* 

### Elucidating the Design Space of Diffusion-Based Generative Models

*Tero Karras*

**Summary:** We argue that the theory and practice of diffusion-based generative models are currently unnecessarily convoluted and seek to remedy the situation by presenting a design space that clearly separates the concrete design choices.

**ArXiv:** [2206.00364](https://arxiv.org/abs/2206.00364), [Local link](Papers%5Cimage_synthesis%5C2206.00364v2_Elucidating%20the%20Design%20Space%20of%20Diffusion-Based.pdf), Hash: 3cee3f625fb2145e042f766e0e53f236, *Added: 2025-01-17* 

### Learning to Generate and Transfer Data with Rectified Flow

*Xingchao Liu, Chengyue Gong, Qiang Liu*

**Summary:** We present rectified flow, a surprisingly simple approach to learning (neural) ordinary differential equation (ODE) models to transport between two empirically observed distributions p0 and p1, hence providing a unified solution to generative modeling and domain transfer, among various other tasks involving distribution transport.

**ArXiv:** [2209.03003](https://arxiv.org/abs/2209.03003), [Local link](Papers%5Cimage_synthesis%5C2209.03003v1_Learning%20to%20Generate%20and%20Transfer%20Data%20with%20Rectified%20Flow.pdf), Hash: fffccc84dd61a9757ddf7330eb396c40, *Added: 2025-01-17* 

### Building Normalizing Flows With Stochastic Interpolants

*Michael S. Albergo, Eric Vanden-Eijnden*

**Summary:** A generative model based on a continuous-time normalizing flow between any pair of base and target probability densities is proposed.

**ArXiv:** [2303.10852](https://arxiv.org/abs/2303.10852), [Local link](Papers%5Cimage_synthesis%5C2209.15571v3_BUILDING%20NORMALIZING%20FLOWS%20WITH%20STOCHASTIC.pdf), Hash: 2cdec2892ce07c5cb72657dcc8a4ef13, *Added: 2025-01-17* 

### Scalable Diffusion Models with Transformers

*William Peebles, Saining Xie*

**Summary:** We explore a new class of diffusion models based on the transformer architecture. We train latent diffusion models of images, replacing the commonly-used U-Net backbone with a transformer that operates on latent patches.

**ArXiv:** [2212.09748](https://arxiv.org/abs/2212.09748), [Local link](Papers%5Cimage_synthesis%5C2212.09748v2_Scalable%20Diffusion%20Models%20with%20Transformers.pdf), Hash: b0fc0e61fd445fd97a00e4a08c1a8bb3, *Added: 2025-01-17* 

### Flow Matching in Latent Space

*Quan Dao, Hao Phung, Binh Nguyen, Anh Tran*

**Summary:** This paper proposes applying flow matching in the latent spaces of pretrained autoencoders to improve computational efficiency and scalability for high-resolution image synthesis.

**ArXiv:** [2307.08698](https://arxiv.org/abs/2307.08698), [Local link](Papers%5Cimage_synthesis%5C2307.08698v1_Flow%20Matching%20in%20Latent%20Space.pdf), Hash: e666d3109498efc40db5f9cce02bf011, *Added: 2025-01-17* 

### Tutorial on Diffusion Models for Imaging and Vision

*Stanley Chan*

**Summary:** This tutorial discusses the essential ideas underlying diffusion models, which have empowered exciting applications in text-to-image generation and text-to-video generation.

**ArXiv:** [2403.18103](https://arxiv.org/abs/2403.18103), [Local link](Papers%5Cimage_synthesis%5C2403.18103v2_Tutorial_on_Diffusion_Models.pdf), Hash: bf04ad5ff022048df73bf8c3608ee803, *Added: 2025-01-17* 

### Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction

*Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, Liwei Wang*

**Summary:** We present Visual AutoRegressive modeling (V AR), a new generation paradigm that redefines the autoregressive learning on images as coarse-to-fine "next-scale prediction" or "next-resolution prediction", diverging from the standard raster-scan "next-token prediction". This simple, intuitive methodology allows autoregressive (AR) transformers to learn visual distributions fast and can generalize well.

**ArXiv:** [2404.02905](https://arxiv.org/abs/2404.02905), [Local link](Papers%5Cimage_synthesis%5C2404.02905.pdf), Hash: 4016f04c122748c124df09fe18b68492, *Added: 2025-01-17* 

### Inference-Time Scaling for Diffusion Models beyond Scaling Denoising Steps

*Nanye Ma, Shangyuan Tong, Haolin Jia, Hexiang Hu, Yu-Chuan Su, Mingda Zhang, Xuan Yang, Yandong Li, Tommi Jaakkola, Xuhui Jia, Saining Xie*

**Summary:** This work explores the inference-time scaling behavior of diffusion models beyond increasing denoising steps, revealing that increasing inference-time compute leads to substantial improvements in the quality of samples generated by diffusion models.

**ArXiv:** [2501.09732](https://arxiv.org/abs/2501.09732), [Local link](Papers%5Cimage_synthesis%5C2501.09732v1_nference-Time%20Scaling%20for%20Diffusion%20Models.pdf), Hash: 07fc33767c022b190649d1144e7a2025, *Added: 2025-01-17* 

### Flow Matching for Generative Modeling

*Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, Matt Le*

**Summary:** We introduce Flow Matching (FM), a simulation-free approach for training Continuous Normalizing Flows (CNFs) based on regressing vector fields of fixed conditional probability paths. This allows for the training of CNFs at unprecedented scale and opens up new possibilities such as using Optimal Transport displacement interpolation to define the conditional probability paths, leading to faster training and better generalization.

**ArXiv:** [2209.15571](https://arxiv.org/abs/2209.15571), [Local link](Papers%5Cimage_synthesis%5C3719_flow_matching_for_generative_m.pdf), Hash: 5452cb6c832015424b6e7774cf8b00a9, *Added: 2025-01-17* 

## Quantization Techniques in AI/ML Research Papers

### DeepShift: Towards Multiplication-Less Neural Networks

*Mostafa Elhoushi, Zihao Chen, Farhan Shafig, Ye Henry Tian, Joey Yiwei Li*

**Summary:** The high computation, memory, and power budgets of inferring convolutional neural networks (CNNs) are major bottlenecks of model deployment to edge computing platforms, e.g., mobile devices and IoT.

**ArXiv:** [1905.13298](https://arxiv.org/abs/1905.13298), [Local link](Papers%5Cquantization%5C1905.13298_deepshift.pdf), Hash: 94eeb0e12a131bbd5e31cec15e40b281, *Added: 2025-01-17* 

### AdderNet: Do We Really Need Multiplications in Deep Learning?

*Hanting Chen, Yunhe Wang, Chunjing Xu, Boxin Shi, Chao Xu, Qi Tian, Chang Xu*

**Summary:** Adder Networks (AdderNets) trade massive multiplications in deep neural networks, especially convolutional neural networks (CNNs), for much cheaper additions to reduce computation costs.

**ArXiv:** [1912.13200](https://arxiv.org/abs/1912.13200), [Local link](Papers%5Cquantization%5C1912.13200.pdf), Hash: 8567dcb1010414feaac0223952396917, *Added: 2025-01-17* 

### VS-Q UANT: PER-VECTOR SCALED QUANTIZATION FOR ACCURATE LOW-PRECISION NEURAL NETWORK INFERENCE

*Steve Dai, Rangharajan Venkatesan, Haoxing Ren, Brian Zimmer, William J. Dally, Brucek Khailany*

**Summary:** Quantization enables efficient acceleration of deep neural networks by reducing model memory footprint and exploiting low-cost integer math hardware units.

**ArXiv:** [2102.04503](https://arxiv.org/abs/2102.04503), [Local link](Papers%5Cquantization%5C2102.04503v1_per_vector_quant.pdf), Hash: b7fdb71333c901b39ecfc058d564e105, *Added: 2025-01-17* 

### LLM.int8() : 8-bit Matrix Multiplication for Transformers at Scale

*Tim Dettmers, Mike Lewis, Younes Belkada, Luke Zettlemoyer*

**Summary:** Large language models have been widely adopted but require significant GPU memory for inference. We develop a procedure for Int8 matrix multiplication for feed-forward and attention projection layers in transformers, which cut the memory needed for inference by half while retaining full precision performance.

**ArXiv:** [2208.07339](https://arxiv.org/abs/2208.07339), [Local link](Papers%5Cquantization%5C2208.07339.pdf), Hash: cde5fd566b995573932664907c0eea57, *Added: 2025-01-17* 

### GPTQ: A Curate Post-Training Quantization for Generative Pre-Trained Transformers

*Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh*

**Summary:** This paper proposes GPTQ, a new one-shot weight quantization method based on approximate second-order information, that is both highly-accurate and highly-efficient. It reduces the bitwidth down to 3 or 4 bits per weight for large GPT models with negligible accuracy degradation.

**ArXiv:** [2210.17323](https://arxiv.org/abs/2210.17323), [Local link](Papers%5Cquantization%5C2210.17323v2_GPTQ.pdf), Hash: 1188fd9e7933a11cf51b835ac07aaa13, *Added: 2025-01-17* 

### Multiplication-Free Transformer Training via Piecewise Affine Operations

*Atli Kosson, Martin Jaggi*

**Summary:** Multiplications are responsible for most of the computational cost involved in neural network training and inference. Recent research has thus looked for ways to reduce the cost associated with them.

**ArXiv:** [2305.17190](https://arxiv.org/abs/2305.17190), [Local link](Papers%5Cquantization%5C2305.17190v2_mulfree.pdf), Hash: 269d214383aa4d76bb0d948da821b526, *Added: 2025-01-17* 

### A WQ: Activation-aware Weight Quantization for LLM Compression and Acceleration

*Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Xingyu Dang, Chuang Gan, Song Han*

**Summary:** Large language models (LLMs) have shown excellent performance on various tasks, but the astronomical model size raises the hardware barrier for serving and slows down token generation. This paper proposes Activation-aware Weight Quantization (AWQ), a hardware-friendly approach for LLM low-bit weight-only quantization based on observing that weights are not equally important.

**ArXiv:** [2306.00978](https://arxiv.org/abs/2306.00978), [Local link](Papers%5Cquantization%5C2306.00978_AWQ.pdf), Hash: e37ecfa8af69a03b4e126934caf4cee3, *Added: 2025-01-17* 

### BitNet: Scaling 1-bit Transformers for Large Language Models

*Hongyu Wang, Shuming Ma, Li Dong, Shaohan Huang, Huaijie Wang, Lingxiao Ma, Fan Yang, Ruiping Wang, Yi Wu, Furu Wei*

**Summary:** BitNet is a scalable and stable 1-bit Transformer architecture designed for large language models, which substantially reduces memory footprint and energy consumption while maintaining competitive performance.

**ArXiv:** [2310.11453](https://arxiv.org/abs/2310.11453), [Local link](Papers%5Cquantization%5C2310.11453_bitnet.pdf), Hash: 2ad51decc58a80910bbd05b8084c676a, *Added: 2025-01-17* 

### FP6-LLM: Efficiently Serving Large Language Models Through FP6-Centric Algorithm-System Co-Design

*Haojun Xia, Zhen Zheng, Xiaoxia Wu, Shiyang Chen, Zhewei Yao, Stephen Youn, Arash Bakhtiari, Michael Wyatt, Donglin Zhuang, Zhongzhu Zhou, Olatunji Ruwase, Yuxiong He, Shuaiwen Leon Song*

**Summary:** Six-bit quantization (FP6) can effectively reduce the size of large language models (LLMs) and preserve the model quality consistently across varied applications. However, existing systems do not provide Tensor Core support for FP6 quantization and struggle to achieve practical performance improvements during LLM inference.

**ArXiv:** [2401.14112](https://arxiv.org/abs/2401.14112), [Local link](Papers%5Cquantization%5C2401.14112v2_fp6_llm.pdf), Hash: 7dbfccd4df499b1e6b6fd59857fe094e, *Added: 2025-01-17* 

### The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits

*Shuming Ma, Hongyu Wang, Lingxiao Ma, Lei Wang, Wenhui Wang, Shaohan Huang, Li Dong, Ruiping Wang, Jilong Xue, Furu Wei*

**Summary:** Recent research, such as BitNet [ WMD+23], is paving the way for a new era of 1- bit Large Language Models (LLMs). In this work, we introduce a 1-bit LLM variant, namely BitNet b1.58 , in which every single parameter (or weight) of the LLM is ternary {-1, 0, 1}. It matches the full-precision (i.e., FP16 or BF16) Transformer LLM with the same model size and training tokens in terms of both perplexity and end-task performance, while being significantly more cost-effective in terms of latency, memory, throughput, and energy consumption.

**ArXiv:** [2402.17764](https://arxiv.org/abs/2402.17764), [Local link](Papers%5Cquantization%5C2402.17764_1_58bit.pdf), Hash: 45f0b9c45de94bd864bd260cc10f40eb, *Added: 2025-01-17* 

### SpinQuant: LLM Quantization with Learned Rotations

*Zechun Liu, Changsheng Zhao, Igor Fedorov Bilge Soran Dhruv Choudhary, Raghuraman Krishnamoorthi, Vikas Chandra, Yuandong Tian, Tijmen Blankevoort*

**Summary:** SpinQuant, a novel approach that incorporates learned rotation matrices for optimal quantized network accuracy, narrows the accuracy gap on zero-shot reasoning tasks with full precision to merely 2.9 points on the LLaMA-27B model, surpassing LLM-QAT by 19.1 points and SmoothQuant by 25.0 points.

**ArXiv:** [2405.16406](https://arxiv.org/abs/2405.16406), [Local link](Papers%5Cquantization%5C2405.16406v3_spinquant.pdf), Hash: e3bd432556e1f3aef9b8a13a134483f3, *Added: 2025-01-17* 

### Scalable MatMul-free Language Modeling

*Rui-Jie Zhu, Yu Zhang, Ethan Sifferman, Tyler Sheaves, Yiqiao Wang, Dustin Richmond, Peng Zhou, Jason K. Eshraghian*

**Summary:** Matrix multiplication (MatMul) typically dominates the overall computational cost of large language models (LLMs). This cost only grows as LLMs scale to larger embedding dimensions and context lengths. In this work, we show that MatMul operations can be completely eliminated from LLMs while maintaining strong performance at billion-parameter scales.

**ArXiv:** [2406.02528](https://arxiv.org/abs/2406.02528), [Local link](Papers%5Cquantization%5C2406.02528v1_matmul_free.pdf), Hash: dc43e3bf03f0f3882f1e15b331027f19, *Added: 2025-01-17* 

### ShiftAddLLM: Accelerating Pretrained LLMs via Post-Training Multiplication-Less Reparameterization

*Haoran You, Yipin Guo, Yichao Fu, Wei Zhou, Huihong Shi, Xiaofan Zhang, Souvik Kundu, Amir Yazdanbakhsh, Yingyan (Celine) Lin*

**Summary:** Large language models (LLMs) face challenges when deployed on resource-constrained devices due to their extensive parameters and reliance on dense multiplications. Shift-and-add reparameterization offers a solution by replacing costly multiplications with hardware-friendly primitives in both the attention and multi-layer perceptron (MLP) layers of an LLM. The authors propose accelerating pretrained LLMs through post-training shift-and-add reparameterization, creating efficient multiplication-free models, dubbed ShiftAddLLM.

**ArXiv:** [2406.05981](https://arxiv.org/abs/2406.05981), [Local link](Papers%5Cquantization%5C2406.05981v3_ShiftAddLLM.pdf), Hash: a0824c77d773f1e6438ac1bed7fde146, *Added: 2025-01-17* 

### QTIP: Quantization with Trellises and Incoherence

*Albert Tseng, Qingyao Sun, David Hou, Christopher De Sa*

**Summary:** Post-training quantization (PTQ) reduces the memory footprint of LLMs by quantizing weights to low-precision datatypes. Since LLM inference is usually memory-bound, PTQ methods can improve inference throughput.

**ArXiv:** [2406.11235](https://arxiv.org/abs/2406.11235), [Local link](Papers%5Cquantization%5C2406.11235v3_QTIP_Quantization%20with%20Trellises%20and%20Incoherence.pdf), Hash: 82d79aceac9594cae0b01c197081f74c, *Added: 2025-01-17* 

### VPTQ: Extreme Low-bit Vector Post-Training Quantization for Large Language Models

*Yifei Liu, Jicheng Wen, Yang Wang, Shengyu Ye, Li Lyna Zhang, Ting Cao, Cheng Li, Mao Yang*

**Summary:** We introduce Vector Post-Training Quantization (VPTQ) for extremely low-bit quantization of Large Language Models, using Second-Order Optimization to guide our quantization algorithm design and refining weights with Channel-Independent Second-Order Optimization.

**ArXiv:** [2409.17066](https://arxiv.org/abs/2409.17066), [Local link](Papers%5Cquantization%5C2409.17066v2_VPTQ.pdf), Hash: af8ff48a50c92f2320a2c1cbbafe431e, *Added: 2025-01-17* 

### How Numerical Precision Affects Mathematical Reasoning Capabilities of LLMs

*Guhao Feng, Kai Yang, Yuntian Gu, Xinyue Ai, Shengjie Luo, Jiacheng Sun, Di He, Zhenguo Li, Liwei Wang*

**Summary:** This paper conducts a theoretical analysis of LLMs' mathematical abilities, focusing on their arithmetic performance and identifying numerical precision as a key factor influencing their effectiveness in mathematical tasks.

**ArXiv:** [2410.13857](https://arxiv.org/abs/2410.13857), [Local link](Papers%5Cquantization%5C2410.13857v1_How%20Numerical%20Precision%20Affects%20Mathematical%20Reasoning%20Capabilities%20of%20LLMs.pdf), Hash: 9260970ce466007a7107a47b631a36ca, *Added: 2025-01-17* 

### Low-Bit Quantization Favors Undertrained LLMs: Scaling Laws for Quantized LLMs with 100T Training Tokens

*Xu Ouyang, Tao Ge, Thomas Hartvigsen, Zhisong Zhang, Haitao Mi, Dong Yu*

**Summary:** We reveal that low-bit quantization favors undertrained large language models (LLMs) by observing that models with larger sizes or fewer training tokens experience less quantization-induced degradation (QiD) when applying low-bit quantization, whereas smaller models with extensive training tokens suffer significant QiD.

**ArXiv:** [2411.17691](https://arxiv.org/abs/2411.17691), [Local link](Papers%5Cquantization%5C2411.17691v2_Low-Bit%20Quantization%20Favors%20Undertrained%20LLMs.pdf), Hash: d7be68b5834160c873d2c83e49593aeb, *Added: 2025-01-17* 

### Least Squares Quantization in PCM

*S. P. Lloyd*

**Summary:** This paper derives necessary conditions that the quanta and associated quantization intervals of an optimum finite quantization scheme must satisfy, based on minimizing average quantization noise power.

**ArXiv:** N/A, [Local link](Papers%5Cquantization%5Clloyd1982_least_squares_quant.pdf), Hash: e8cfe41ea5354727e98fe6ffd6d7e2e1, *Added: 2025-01-17* 

### Stationary and Trellis Encoding for IID Sources and Simulation

*Mark Z. Mao, Robert M. Gray*

**Summary:** Necessary conditions for asymptotically optimal sliding-block or stationary codes for source coding and rate-constrained simulation are presented and applied to a design technique for trellis-encoded source coding and rate-constrained simulation of memoryless sources.

**ArXiv:** N/A, [Local link](Papers%5Cquantization%5Cmao2010.pdf), Hash: 343f8440527dc7b3680a11ef171711d0, *Added: 2025-01-17* 

### Trellis Coded Quantization of Memoryless and Gauss-Markov Sources

*Michael W. Marcellin, Thomas R. Fischer*

**Summary:** Exploiting the duality between modulation for digital communications and source coding, trellis coded quantization (TCQ) is developed and applied to the encoding of memoryless and Gauss-Markov sources.

**ArXiv:** N/A, [Local link](Papers%5Cquantization%5Cmarcellin1990_Trellis%20Coded%20Quantization%20of%20Memoryless%20and.pdf), Hash: 1e5018079047212dc00efddce62d3f0e, *Added: 2025-01-17* 

### ShiftAddNet: A Hardware-Inspired Deep Network

*Haoran You, Xiaohan Chen, Yongan Zhang, Chaojian Li, Sicheng Li, Zihao Liu, Zhangyang Wang, Yingyan Lin*

**Summary:** This paper introduces ShiftAddNet, a deep network inspired by energy-efficient hardware implementation that uses bit-shift and additive weight layers instead of multiplications.

**ArXiv:** [1905.13298](https://arxiv.org/abs/1905.13298), [Local link](Papers%5Cquantization%5CNeurIPS-2020-shiftaddnet-a-hardware-inspired-deep-network-Paper.pdf), Hash: bae3561ed2129c272887af8ce50df626, *Added: 2025-01-17* 

### OCP Microscaling Formats (MX) Specification

*Bita Darvish Rouhani, Nitin Garegrat, Tom Savell, Ankit More, Kyung-Nam Han, Ritchie Zhao, Mathew Hall, Jasmine Klar, Eric Chung, Yuan Yu, Microsoft, Michael Schulte, Ralph Wittig, AMD, Ian Bratt, Nigel Stephens, Jelena Milanovic, John Brothers, Arm, Pradeep Dubey, Marius Cornea, Alexander Heinecke, Andres Rodriguez, Martin Langhammer, Intel, Summer Deng, Maxim Naumov, Meta, Paulius Micikevicius, Michael Siu, NVIDIA, Colin Verrilli, Qualcomm*

**Summary:** This document specifies the Open Compute Project (OCP) Microscaling Formats Specification. It defines microscaling formats, including FP8, FP6, FP4, INT8, and E8M0 scale data types.

**ArXiv:** N/A, [Local link](Papers%5Cquantization%5COCP_Microscaling%20Formats%20%28MX%29%20v1.0%20Spec_Final.pdf), Hash: 9a52e4a8d4a28b49ecef2a3a3f918337, *Added: 2025-01-17* 

## Unsorted Papers

### Distilling the Knowledge in a Neural Network

*Geoffrey Hinton, Oriol Vinyals, Jeff Dean*

**Summary:** A very simple way to improve the performance of almost any machine learning algorithm is to train many different models on the same data and then to average their predictions [3]. Unfortunately, making predictions using a whole ensemble of models is cumbersome and may be too computationally expensive to allow deployment to a large number of users, especially if the individual models are large neural nets.

**ArXiv:** [1503.02531](https://arxiv.org/abs/1503.02531), [Local link](Papers%5C1503.02531v1_Distilling%20the%20Knowledge%20in%20a%20Neural%20Network.pdf), Hash: 656af4f7ffb9c473df4ebc4d50d3712a, *Added: 2025-01-17* 

### Using the Output Embedding to Improve Language Models

*Ofr Press, Lior Wolf*

**Summary:** We study the topmost weight matrix of neural network language models. We show that this matrix constitutes a valid word embedding. When training language models, we recommend tying the input embedding and this output embedding.

**ArXiv:** [1608.05859v3](https://arxiv.org/abs/1608.05859v3), [Local link](Papers%5C1608.05859v3_Using%20the%20Output%20Embedding%20to%20Improve%20Language%20Models.pdf), Hash: 78e309341d805929595bda2b7ed1cf63, *Added: 2025-01-17* 

### Trained Ternary Quantization

*Chenzhuo Zhu, Song Han, Huizi Mao, William J. Dally*

**Summary:** Deep neural networks are widely used in machine learning applications. However, the deployment of large neural networks models can be difficult to deploy on mobile devices with limited power budgets. To solve this problem, we propose Trained Ternary Quantization (TTQ), a method that can reduce the precision of weights in neural networks to ternary values.

**ArXiv:** [1612.01064](https://arxiv.org/abs/1612.01064), [Local link](Papers%5C1612.01064v3.pdf), Hash: 3c9b0ed9eca03d76395f2c6afdf3e219, *Added: 2025-01-17* 

### Proximal Policy Optimization Algorithms

*John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov*

**Summary:** We propose a new family of policy gradient methods for reinforcement learning, which al- ternate between sampling data through interaction with the environment, and optimizing a surrogate objective function using stochastic gradient ascent.

**ArXiv:** [1707.06347](https://arxiv.org/abs/1707.06347), [Local link](Papers%5C1707.06347v2_PPO.pdf), Hash: 9e0725fc34e41616c0faab684513e464, *Added: 2025-01-17* 

### Averaging Weights Leads to Wider Optima and Better Generalization

*Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov, Dmitry Vetrov, Andrew Gordon Wilson*

**Summary:** Deep neural networks are typically trained by optimizing a loss function with an SGD variant, in conjunction with a decaying learning rate, until convergence. We show that simple averaging of multiple points along the trajectory of SGD, with a cyclical or constant learning rate, leads to better generalization than conventional training.

**ArXiv:** [1803.05407](https://arxiv.org/abs/1803.05407), [Local link](Papers%5C1803.05407v3_Stochastic_Weight_Averaging.pdf), Hash: ee58cbd9aa002307de991da7560cb979, *Added: 2025-01-17* 

### World Models

*David Ha, Jürgen Schmidhuber*

**Summary:** We explore building generative neural network models of popular reinforcement learning environments.

**ArXiv:** [1803.10122](https://arxiv.org/abs/1803.10122), [Local link](Papers%5C1803.10122.pdf), Hash: 9df6f092e3faf2105c0d87e5477d5c67, *Added: 2025-01-17* 

### SqueezeNext: Hardware-Aware Neural Network Design

*Amir Gholami, Kiseok Kwon, Bichen Wu, Zizheng Tai, Xiangyu Yue, Peter Jin, Sicheng Zhao, Kurt Keutzer*

**Summary:** One of the main barriers for deploying neural networks on embedded systems has been large memory and power consumption of existing neural networks. In this work, we introduce SqueezeNext, a new family of neural network architectures whose design was guided by considering previous architectures such as SqueezeNet, as well as by simulation results on a neural network accelerator.

**ArXiv:** [1803.10615](https://arxiv.org/abs/1803.10615), [Local link](Papers%5C1803.10615v2_squeezenext.pdf), Hash: ad6c65774f83c471c4529189e69999c6, *Added: 2025-01-17* 

### Deep k-Means: Re-Training and Parameter Sharing with Harder Cluster Assignments for Compressing Deep Convolutions

*Junru Wu, Yue Wang, Zhenyu Wu, Zhangyang Wang, Ashok Veeraraghavan, Yingyan Lin*

**Summary:** We proposed a simple yet effective scheme for compressing convolutions though applying k-means clustering on the weights, compression is achieved through weight-sharing, by only recording Kcluster centers and weight assignment indexes. We additionally propose an improved set of metrics to estimate energy consumption of CNN hardware implementations.

**ArXiv:** [1806.09228](https://arxiv.org/abs/1806.09228), [Local link](Papers%5C1806.09228v1_palettization.pdf), Hash: 3988217769b90c5ec8a068684e558d8c, *Added: 2025-01-17* 

### Universal Transformers

*Mostafa Dehghani, Stephan Gouws, Oriol Vinyals, Jakob Uszkoreit, Łukasz Kaiser*

**Summary:** Recurrent neural networks (RNNs) sequentially process data by updating their state with each new data point, and have long been the de facto choice for sequence modeling tasks. However, their inherently sequential computation makes them slow to train.

**ArXiv:** [1807.03819](https://arxiv.org/abs/1807.03819), [Local link](Papers%5C1807.03819v3_UNIVERSAL%20TRANSFORMERS.pdf), Hash: 875cb71a8fec077ef9c33174ff0f04ae, *Added: 2025-01-17* 

### Improving Neural Network Quantization without Retraining using Outlier Channel Splitting

*Ritchie Zhao, Yuwei Hu, Jordan Dotzel, Christopher De Sa, Zhiru Zhang*

**Summary:** Quantization can improve the execution latency and energy efficiency of neural networks on both commodity GPUs and specialized accelerators.

**ArXiv:** [1901.09504](https://arxiv.org/abs/1901.09504), [Local link](Papers%5C1901.09504v3_outlier%20channel%20splitting.pdf), Hash: 745960627406b452459035a5285e7824, *Added: 2025-01-17* 

### Learned Step Size Quantization

*Steven K. Esser, Jeffrey L. McKinstry, Deepika Bablani, Rathinakumar Appuswamy, Dharmendra S. Modha*

**Summary:** This paper presents Learned Step Size Quantization, a method for training deep networks with low precision operations at inference time that achieves high accuracy on the ImageNet dataset using models with weights and activations quantized to 2-, 3- or 4-bits of precision.

**ArXiv:** [1902.08153](https://arxiv.org/abs/1902.08153), [Local link](Papers%5C1902.08153v3_good_paper_qat.pdf), Hash: 601fe34a0b921f764bd1c091b25b024b, *Added: 2025-01-17* 

### All You Need Is a Few Shifts: Designing Efficient Convolutional Neural Networks for Image Classification

*Weijie Chen, Di Xie, Yuan Zhang, Shiliang Pu*

**Summary:** Shift operation is an efficient alternative over depthwise separable convolution. However, it is still bottlenecked by its implementation manner, namely memory movement. To put this direction forward, a new and novel basic component named Sparse Shift Layer (SSL) is introduced in this paper to construct efficient convolutional neural networks.

**ArXiv:** [1903.05285](https://arxiv.org/abs/1903.05285), [Local link](Papers%5C1903.05285.pdf), Hash: b9e14495a718204775dd13877ae5c5a5, *Added: 2025-01-17* 

### Character Region Awareness for Text Detection

*Youngmin Baek, Bado Lee, Dongyoon Han, Sangdoo Yun, Hwalsuk Lee*

**Summary:** This paper proposes a new scene text detection method that effectively detects text areas by exploring each character and affinity between characters, significantly outperforming state-of-the-art detectors on benchmarks with highly curved texts.

**ArXiv:** [1904.01941](https://arxiv.org/abs/1904.01941), [Local link](Papers%5C1904.01941v1_craft_text_detection.pdf), Hash: c78b0a8bcb1f041fe659c25cd16832f9, *Added: 2025-01-17* 

### Searching for MobileNetV3

*Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam*

**Summary:** We present the next generation of MobileNets based on a combination of complementary search techniques as well as a novel architecture design.

**ArXiv:** [1905.02244](https://arxiv.org/abs/1905.02244), [Local link](Papers%5C1905.02244v5_mobilenetv3.pdf), Hash: 6f29ad563a89d1086adf5e7f990c6e0a, *Added: 2025-01-17* 

### EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks

*Mingxing Tan, Quoc V. Le*

**Summary:** Convolutional Neural Networks (ConvNets) are commonly developed at a fixed resource budget, and then scaled up for better accuracy if more resources are available. In this paper, we systematically study model scaling and identify that carefully balancing network depth, width, and resolution can lead to better performance.

**ArXiv:** [1905.11946](https://arxiv.org/abs/1905.11946), [Local link](Papers%5C1905.11946_efficientnet.pdf), Hash: a6491e929ceb9af0db1f063a6a69fb61, *Added: 2025-01-17* 

### Large Memory Layers with Product Keys

*Guillaume Lample, Alexandre Sablayrolles, Marc'Aurelio Ranzato, Ludovic Denoyer, Herv'e Jegou*

**Summary:** This paper introduces a structured memory which can be easily integrated into a neural network. The memory is very large by design and significantly increases the capacity of the architecture, by up to a billion parameters with a negligible computational overhead.

**ArXiv:** [1907.05242](https://arxiv.org/abs/1907.05242), [Local link](Papers%5C1907.05242v2_Large%20Memory%20Layers%20with%20Product%20Keys.pdf), Hash: ef23126232c6b56adc91789c05a6ba0d, *Added: 2025-01-17* 

### ADDITIVE POWERS-OF-TWO QUANTIZATION: AN EFFICIENT NON-UNIFORM DISCRETIZATION FOR NEURAL NETWORKS

*Yuhang Li, Xin Dong, Wei Wang*

**Summary:** We propose Additive Powers-of-Two (APoT) quantization, an efficient non-uniform quantization scheme for the bell-shaped and long-tailed distribution of weights and activations in neural networks.

**ArXiv:** [1909.13144](https://arxiv.org/abs/1909.13144), [Local link](Papers%5C1909.13144v2_expcoding.pdf), Hash: 614a358832156d33df4e2f86e8360e2c, *Added: 2025-01-17* 

### Scaling Laws for Neural Language Models

*Jared Kaplan, Johns Hopkins University, OpenAI, jaredk@jhu.edu, Sam McCandlish, OpenAI, sam@openai.com, Tom Henighan, OpenAI, henighan@openai.com, Tom B. Brown, OpenAI, tom@openai.com, Benjamin Chess, OpenAI, bchess@openai.com, Rewon Child, OpenAI, rewon@openai.com, Scott Gray, OpenAI, scott@openai.com, Alec Radford, OpenAI, alec@openai.com, Jeffrey Wu, OpenAI, jeffwu@openai.com, Dario Amodei, OpenAI, damodei@openai.com*

**Summary:** We study empirical scaling laws for language model performance on the cross-entropy loss. The loss scales as a power-law with model size, dataset size, and the amount of compute used for training, with some trends spanning more than seven orders of magnitude.

**ArXiv:** [2001.08361](https://arxiv.org/abs/2001.08361), [Local link](Papers%5C2001.08361v1_Scaling%20Laws%20for%20Neural%20Language%20Models.pdf), Hash: 66bbfa515aad7a216df52938b9c1ecd8, *Added: 2025-01-17* 

### On Layer Normalization in the Transformer Architecture

*Ruibin Xiong, Yunchang Yang, Di He, Kai Zheng, Shuxin Zheng, Chen Xing, Huishuai Zhang, Yanyan Lan, Liwei Wang, Tie-Yan Liu*

**Summary:** The paper studies the importance of layer normalization in the Transformer architecture and shows that using Pre-LN Transformers without a warm-up stage can lead to comparable results with less training time and hyper-parameter tuning.

**ArXiv:** [2002.04745](https://arxiv.org/abs/2002.04745), [Local link](Papers%5C2002.04745v2_On%20Layer%20Normalization%20in%20the%20Transformer%20Architecture.pdf), Hash: d7c55fb6bd5fdb5d14364b61b7bef09d, *Added: 2025-01-17* 

### GLU Variants Improve Transformer Feed Forward Networks

*Noam Shazeer*

**Summary:** Variations on Gated Linear Units (GLU) are tested in the feed-forward sublayers of the Transformer sequence-to-sequence model, and some variants yield quality improvements over ReLU or GELU activations.

**ArXiv:** [2002.05202v1](https://arxiv.org/abs/2002.05202v1), [Local link](Papers%5C2002.05202v1_GLU%20Variants%20Improve%20Transformer.pdf), Hash: 8781c29b4f58681ea17211c4e9e5e3f9, *Added: 2025-01-17* 

### A Neural Scaling Law from the Dimension of the Data Manifold

*Utkarsh Sharma, Jared Kaplan*

**Summary:** When data is plentiful, the loss achieved by well-trained neural networks scales as a power-law L/N^d in the number of network parameters N. This empirical scaling law holds for a wide variety of data modalities, and may persist over many orders of magnitude.

**ArXiv:** [2004.10802](https://arxiv.org/abs/2004.10802), [Local link](Papers%5C2004.10802v1_A%20Neural%20Scaling%20Law%20from%20dimension.pdf), Hash: 9417666444dc9b259032695d5f3d1ea3, *Added: 2025-01-17* 

### Language Models are Few-Shot Learners

*Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, Dario Amodei*

**Summary:** Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by pre-training on a large corpus of text followed by fine-tuning on a specific task. While typically task-agnostic in architecture, this method still requires task-specific fine-tuning datasets of thousands or tens of thousands of examples. By contrast, humans can generally perform a new language task from only a few examples or from simple instructions – something which current NLP systems still largely struggle to do.

**ArXiv:** [2005.14165](https://arxiv.org/abs/2005.14165), [Local link](Papers%5C2005.14165v4_Language%20Models%20are%20Few-Shot%20Learners.pdf), Hash: f1798481561d9495486bc007180aa998, *Added: 2025-01-17* 

### AdderSR: Towards Energy Efficient Image Super-Resolution

*Dehua Song, Yunhe Wang, Hanting Chen, Chang Xu, Chunjing Xu, Dacheng Tao*

**Summary:** This paper studies the single image super-resolution problem using adder neural networks (AdderNets). Compared with convolutional neural networks, AdderNets utilize additions to calculate the output features thus avoid massive energy consumptions of conventional multiplications.

**ArXiv:** [2009.08891](https://arxiv.org/abs/2009.08891), [Local link](Papers%5C2009.08891.pdf), Hash: ed2b2c74df22b3359386cf86ce266683, *Added: 2025-01-17* 

### ANIMAGE IS WORTH 16X16 W ORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE

*Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby*

**Summary:** We show that the reliance on CNNs in vision is not necessary and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks.

**ArXiv:** [2010.11929](https://arxiv.org/abs/2010.11929), [Local link](Papers%5C2010.11929.pdf), Hash: 7f1ba6e93e08ec7fc3f4fa1c8a255ac0, *Added: 2025-01-17* 

### Scaling Laws for Autoregressive Generative Modeling

*Tom Henighan, Jared Kaplan, Mor Katz, Mark Chen, Christopher Hesse, Jacob Jackson, Heewoo Jun, Tom B. Brown, Prafulla Dhariwal, Scott Gray, Chris Hallacy, Benjamin Mann, Alec Radford, Aditya Ramesh, Nick Ryder, Daniel M. Ziegler, John Schulman, Dario Amodei, Sam McCandlish*

**Summary:** We identify empirical scaling laws for the cross-entropy loss in four domains: generative image modeling, video modeling, multimodal image $text models, and mathematical problem solving. In all cases autoregressive Transformers smoothly improve in performance as model size and compute budgets increase, following a power-law plus constant scaling law.

**ArXiv:** [2010.14701](https://arxiv.org/abs/2010.14701), [Local link](Papers%5C2010.14701v2_Scaling%20Laws%20for%20Autoregressive%20Generative%20Modeling.pdf), Hash: 40a9d361e2de911dc404de33743b9b49, *Added: 2025-01-17* 

### Sparsity in Deep Learning: Pruning and growth for efficient inference and training in neural networks

*Torsten Hoefler, Dan Alistarh, Tal Ben-Nun, Nikoli Dryden, Alexandra Peste*

**Summary:** The growing energy and performance costs of deep learning have driven the community to reduce the size of neural networks by selectively pruning components. Similarly to their biological counterparts, sparse networks generalize just as well, if not better than, the original dense networks. Sparsity can reduce the memory footprint of regular networks to fit mobile devices, as well as shorten training time for ever growing networks.

**ArXiv:** [2102.00554](https://arxiv.org/abs/2102.00554), [Local link](Papers%5C2102.00554v1_sparsity.pdf), Hash: 2220739fe6d99860c784327d505b186d, *Added: 2025-01-17* 

### Explaining Neural Scaling Laws

*Yasaman Bahri, Ethan Dyer, Jared Kaplan, Jaehoon Lee, Utkarsh Sharma*

**Summary:** The population loss of trained deep neural networks often follows precise power-law scaling relations with either the size of the training dataset or the number of parameters in the network. We propose a theory that explains the origins of and connects these scaling laws.

**ArXiv:** [2102.06701](https://arxiv.org/abs/2102.06701), [Local link](Papers%5C2102.06701v2_Explaining%20Neural%20Scaling%20Laws.pdf), Hash: ba5fb65cbb4bcbeadbfb02fb2fcb70fd, *Added: 2025-01-17* 

### Network Quantization with Element-wise Gradient Scaling

*Junghyup Lee, Dohyung Kim, Bumsub Ham*

**Summary:** Network quantization aims at reducing bit-widths of weights and/or activations, particularly important for implementing deep neural networks with limited hardware resources.

**ArXiv:** [2104.00903](https://arxiv.org/abs/2104.00903), [Local link](Papers%5C2104.00903v1_STE_Optimization_EWGS.pdf), Hash: b54c7ede8c2bafc99ef592b057b66e9d, *Added: 2025-01-17* 

### LeViT: a Vision Transformer in ConvNet's Clothing for Faster Inference

*Benjamin Graham, Alaaeldin El-Nouby, Hugo Touvron, Pierre Stock, Armand Joulin, Hervé Jégou, Matthijs Douze*

**Summary:** We design a family of image classification architectures that optimize the trade-off between accuracy and efficiency in a high-speed regime, proposing LeVIT: a hybrid neural network for fast inference image classification.

**ArXiv:** [2104.01136](https://arxiv.org/abs/2104.01136), [Local link](Papers%5C2104.01136v2_levit_light_weight_vision.pdf), Hash: c74f4b03484329d133a1188ab9c8bd4b, *Added: 2025-01-17* 

### ROFORMER: Enhanced Transformer with Rotary Position Embedding

*Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, Yunfeng Liu*

**Summary:** Position encoding recently has shown effective in the transformer architecture. It enables valuable supervision for dependency modeling between elements at different positions of the sequence.

**ArXiv:** [2104.09864](https://arxiv.org/abs/2104.09864), [Local link](Papers%5C2104.09864v5_Enhanced%20Transformer%20with%20Rotary%20Position%20Embedding.pdf), Hash: 17b549c747134a1e6e4dba5a51cadf5f, *Added: 2025-01-17* 

### DKM: Differentiable k-Means Clustering Layer for Neural Network Compression

*Minsik Cho, Keivan Alizadeh-Vahid, Saurabh Adya, Mohammad Rastegari*

**Summary:** This paper proposes a differentiable k-means clustering layer (DKM) for efficient on-device neural network inference by reducing memory requirements and keeping user data on-device.

**ArXiv:** [2108.12659v4](https://arxiv.org/abs/2108.12659v4), [Local link](Papers%5C2108.12659v4_palettization.pdf), Hash: 806791b8719a1735064e32741d0db5b7, *Added: 2025-01-17* 

### Primer: Searching for Efficient Transformers for Language Modeling

*David R. So, Wojciech Mankowski, Hanxiao Liu, Zihang Dai, Noam Shazeer, Quoc V. Le*

**Summary:** Large Transformer models have been central to recent advances in natural language processing. The training and inference costs of these models, however, have grown rapidly and become prohibitively expensive. Here we aim to reduce the costs of Transformers by searching for a more efficient variant.

**ArXiv:** [2109.08668](https://arxiv.org/abs/2109.08668), [Local link](Papers%5C2109.08668v2_primer_llm_architecture_search.pdf), Hash: b0b2d5391fbd2c9ce4bdf4a29f18ef67, *Added: 2025-01-17* 

### Chain-of-Thought Prompting Elicits Reasoning in Large Language Models

*Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V . Le, Denny Zhou*

**Summary:** We explore how generating a chain of thought —a series of intermediate reasoning steps—significantly improves the ability of large language models to perform complex reasoning via a simple method called chain-of-thought prompting.

**ArXiv:** [2306.04178](https://arxiv.org/abs/2306.04178), [Local link](Papers%5C2201.11903v6_Chain-of-Thought.pdf), Hash: bb5cb052621074b4426998fcfeee100b, *Added: 2025-01-17* 

### Gradients Without Backpropagation

*Atülım G"unes Baydin, Barak A. Pearlmutter, Don Syme, Frank Wood, Philip Torr*

**Summary:** We present a method to compute gradients based solely on the directional derivative that can be evaluated in a single forward run of the function, entirely eliminating the need for backpropagation in gradient descent.

**ArXiv:** [2202.08587](https://arxiv.org/abs/2202.08587), [Local link](Papers%5C2202.08587v1_Gradients%20without%20Backpropagation.pdf), Hash: daa9ee81ab383dc4d86cec2a94230c86, *Added: 2025-01-17* 

### Rare Gems: Finding Lottery Tickets at Initialization

*Kartik Sreenivasan, Jy-yong Sohn, Liu Yang, Matthew Grindenwalt Nagle, Hongyi Wang, Eric Xing, Kangwook Lee, Dimitris Papailiopoulos*

**Summary:** GEM-MINER finds lottery tickets at initialization that beat current baselines, finding trainable networks up to 19x faster than Iterative Magnitude Pruning (IMP) and achieving accuracy competitive or better.

**ArXiv:** [2202.12002](https://arxiv.org/abs/2202.12002), [Local link](Papers%5C2202.12002v2_Finding%20Lottery%20Tickets%20at%20Initialization.pdf), Hash: 4823419ab710161dca0fd8515e29582f, *Added: 2025-01-17* 

### Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer

*Greg Yang, Edward J. Hu, Igor Babuschkin, Szymon Sidor, Xiaodong Liu, David Farhi, Nick Ryder, Jakub Pachocki, Weizhu Chen, Jianfeng Gao*

**Summary:** Hyperparameter (HP) tuning in deep learning is an expensive process, prohibitively so for neural networks (NNs) with billions of parameters. We show that, in the recently discovered Maximal Update Parametrization (P), many optimal HPs remain stable even as model size changes. This leads to a new HP tuning paradigm we call Transfer: parametrize the target model in P, tune the HP indirectly on a smaller model and zero-shot transfer them to the full-sized model, i.e., without directly tuning the latter at all.

**ArXiv:** [2203.03466](https://arxiv.org/abs/2203.03466), [Local link](Papers%5C2203.03466v2_Tuning%20Large%20Neural%20Networks%20via%20Zero%20Shot.pdf), Hash: 76aedc789713d175694f0639cdd682a4, *Added: 2025-01-17* 

### STaR: Self-Taught Reasoner Bootstrapping Reasoning With Reasoning

*Eric Zelikman, Yuhuai Wu, Jesse Mu, Noah D. Goodman*

**Summary:** Generating step-by-step 'chain-of-thought' rationales improves language model performance on complex reasoning tasks like mathematics or commonsense question-answering.

**ArXiv:** [2203.14465](https://arxiv.org/abs/2203.14465), [Local link](Papers%5C2203.14465v2_STaR_Bootstrapping%20Reasoning%20With%20Reasoning.pdf), Hash: 954323e019ed7433cea0c9c4d466b810, *Added: 2025-01-17* 

### Training Compute-Optimal Large Language Models

*Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, Katie Millican, George van den Driessche, Bogdan Damoc, Aurelia Guy, Simon Osindero, Karen Simonyan, Erich Elsen, Jack W. Rae, Oriol Vinyals, Laurent Sifre*

**Summary:** Weinvestigatetheoptimalmodelsizeandnumberoftokensfortrainingatransformerlanguagemodelunderagivencomputebudget. Weﬁndthatcurrentlargelanguagemodelsaresigniﬁcantlyunder-trained,aconsequenceoftherecentfocusonscalinglanguagemodelswhilstkeepingtheamountoftrainingdataconstant.

**ArXiv:** [2203.15556](https://arxiv.org/abs/2203.15556), [Local link](Papers%5C2203.15556v1_Training%20Compute-Optimal%20Large%20Language%20Models.pdf), Hash: acb47d6c4ce87a62c9edd0db18610618, *Added: 2025-01-17* 

### Optimal Clipping and Magnitude-aware Differentiation for Improved Quantization-aware Training

*Charbel Sakr, Steve Dai, Rangharajan Venkatesan, Brian Zimmer, William J. Dally, Brucek Khailany*

**Summary:** We propose Optimally Clipped Tensors And Vectors (OCTA-V), a recursive algorithm to determine MSE-optimal clipping scalars for improved quantization-aware training (QAT). We also reveal limitations in common gradient estimation techniques and propose magnitude-aware differentiation as a remedy.

**ArXiv:** [2206.06501](https://arxiv.org/abs/2206.06501), [Local link](Papers%5C2206.06501v1_optimal_clipping.pdf), Hash: 55168ecaa41e3f8fb55f87d69c766b7e, *Added: 2025-01-17* 

### Symbolic Discovery of Optimization Algorithms

*Xiangning Chen, Da Huang, Esteban Real, Kaiyuan Wang, Yao Liu, Hieu Pham, Xuanyi Dong, Thang Luong, Cho-Jui Hsieh, Yifeng Lu, Quoc V. Le*

**Summary:** We present a method to formulate algorithm discovery as program search, and apply it to discover optimization algorithms for deep neural network training.

**ArXiv:** [2302.06675](https://arxiv.org/abs/2302.06675), [Local link](Papers%5C2302.06675v4_Lion_Opitmizer.pdf), Hash: 03ef2f8cf9b9a6b67b177bf857f5f057, *Added: 2025-01-17* 

### TinyStories: How Small Can Language Models Be and Still Speak Coherent English?

*Ronen Eldan, Yuanzhi Li*

**Summary:** This work introduces TinyStories, a synthetic dataset of short stories generated by GPT-3.5 and GPT-4, which can be used to train and evaluate LMs that are much smaller than the state-of-the-art models (below 10 million total parameters) or have much simpler architectures (with only one transformer block), yet still produce fluent and consistent stories.

**ArXiv:** [2305.07759](https://arxiv.org/abs/2305.07759), [Local link](Papers%5C2305.07759v2_TinyStories.pdf), Hash: 7265961c095080d829963ae7897bd0db, *Added: 2025-01-17* 

### Let's Verify Step by Step

*Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, Karl Cobbe*

**Summary:** In this paper, the authors investigate the effectiveness of process supervision compared to outcome supervision for training reliable language models. They find that process supervision significantly outperforms outcome supervision when solving problems from the challenging MATH dataset, achieving a 78% success rate on a representative subset.

**ArXiv:** [2305.20050](https://arxiv.org/abs/2305.20050), [Local link](Papers%5C2305.20050v1_Let%E2%80%99s%20Verify%20Step%20by%20Step.pdf), Hash: 7b7306d1edb567ccf282872ca7cc58f6, *Added: 2025-01-17* 

### MiniLLM: Knowledge Distillation of Large Language Models

*Yuxian Gu, Li Dong, Furu Wei, Minlie Huang*

**Summary:** Knowledge distillation is a promising technique for reducing the high computational demand of large language models. However, previous knowledge distillation methods are primarily applied to white-box classification models or training small models to imitate black-box model APIs like ChatGPT. How to effectively distill the knowledge of white-box LLMs into small models is still under-explored, which becomes more important with the prosperity of open-source LLMs.

**ArXiv:** [2306.08543](https://arxiv.org/abs/2306.08543), [Local link](Papers%5C2306.08543v4_MiniLLM.pdf), Hash: 816fa6bf389ef8449adc866cdf481017, *Added: 2025-01-17* 

### Patch n’ Pack: NaViT, a Vision Transformer for any Aspect Ratio and Resolution

*Mostafa Dehghani, Basil Mustafa, Josip Djolonga, Jonathan Heek, Matthias Minderer, Mathilde Caron, Andreas Steiner, Joan Puigcerver, Robert Geirhos, Ibrahim Alabdulmohsin, Avital Oliver, Piotr Padlewski, Alexey Gritsenko, Mario Lučić, Neil Houlsby*

**Summary:** The ubiquitous and demonstrably suboptimal choice of resizing images to a fixed resolution before processing them with computer vision models has not yet been successfully challenged. However, models such as the Vision Transformer (ViT) offer flexible sequence-based modeling, and hence varying input sequence lengths. We take advantage of this with NaViT(Native Resolution ViT) which uses sequence packing during training to process inputs of arbitrary resolutions and aspect ratios.

**ArXiv:** [2307.06304](https://arxiv.org/abs/2307.06304), [Local link](Papers%5C2307.06304.pdf), Hash: afbbc25f4c7825c575cc757c566cd4f5, *Added: 2025-01-17* 

### PyPIM: Integrating Digital Processing-in-Memory from Microarchitectural Design to Python Tensors

*Orian Leitersdorf, Ronny Ronen, Shahar Kvatinsky*

**Summary:** This paper provides an end-to-end architectural integration of digital memristive PIM from a high-level Python library for tensor operations to the low-level microarchitectural design.

**ArXiv:** [2308.14007](https://arxiv.org/abs/2308.14007), [Local link](Papers%5C2308.14007v2_PyPIM.pdf), Hash: d0c7bde579fc669199b9dbfdb476ecdd, *Added: 2025-01-17* 

### Explaining Grokking through Circuit Efficiency

*Vikrant Varma, Rohin Shah, Zachary Kenton, János Kramár, Ramana Kumar*

**Summary:** We propose that grokking occurs when a task admits both a generalising solution and a memorising solution, where the generalising solution is slower to learn but more efficient, producing larger logits with the same parameter norm.

**ArXiv:** [2309.02390](https://arxiv.org/abs/2309.02390), [Local link](Papers%5C2309.02390v1_grokking.pdf), Hash: 11aa974cb9279a9217167206b42451f4, *Added: 2025-01-17* 

### Training Chain-of-Thought via Latent-Variable Inference

*Du Phan, Matthew D. Hoffman, David Dohan, Sholto Douglas Tuan Anh Le, Aaron Parisi, Pavel Sountsov, Charles Sutton, Sharad Vikram, Rif A. Saurous*

**Summary:** We propose a fine-tuning strategy that tries to maximize the marginal log-likelihood of generating a correct answer using CoT prompting, approximately averaging over all possible rationales.

**ArXiv:** [2304.06743](https://arxiv.org/abs/2304.06743), [Local link](Papers%5C2312.02179v1.pdf), Hash: d1bcfde44450e930723172a74b4bc72b, *Added: 2025-01-17* 

### REFT: Reasoning with Reinforced Fine-Tuning

*Trung Quoc Luong, Xinbo Zhang, Zhanming Jie, Peng Sun, Xiaoran Jin, Hang Li*

**Summary:** Reinforced Fine-Tuning (ReFT) is proposed to enhance the generalizability of learning Large Language Models for reasoning, using math problem-solving as an example. It first warms up the model with Supervised Fine-Tuning and then employs online reinforcement learning to further fine-tune the model.

**ArXiv:** [2401.08967](https://arxiv.org/abs/2401.08967), [Local link](Papers%5C2401.08967v2_REFT.pdf), Hash: 145e3eae6b874df8ba074c18badabd71, *Added: 2025-01-17* 

### No Free Prune: Information-Theoretic Barriers to Pruning at Initialization

*Tanishq Kumar, Kevin Luo, Mark Sellke*

**Summary:** The existence of lottery tickets Frankle & Carbin (2018) at or near initialization raises the tantalizing question of whether large models are necessary in deep learning, or whether sparse networks can be quickly identified and trained without ever training the dense models that contain them. However, efforts to find these sparse subnetworks without training the dense model ('pruning at initialization') have been broadly unsuccessful Frankle et al. (2020b).

**ArXiv:** [2402.01089](https://arxiv.org/abs/2402.01089), [Local link](Papers%5C2402.01089v1_No%20Free%20Prune.pdf), Hash: e6582e9efd12f859cbd8be9953195142, *Added: 2025-01-17* 

### ReLU2Wins: Discovering Efficient Activation Functions for Sparse LLMs

*Zhengyan Zhang, Yixin Song, Guanghui Yu, Xu Han, Yankai Lin, Chaojun Xiao, Chenyang Song, Zhiyuan Liu, Zeyu Mi, Maosong Sun*

**Summary:** We introduce a general method that defines neuron activation through neuron output magnitudes and a tailored magnitude threshold, demonstrating that non-ReLU LLMs also exhibit sparse activation. We propose a systematic framework to examine the sparsity of LLMs from three aspects: the trade-off between sparsity and performance, the predictivity of sparsity, and the hardware affinity. Our results indicate that models employing ReLU2 excel across all three evaluation aspects, highlighting its potential as an efficient activation function for sparse LLMs.

**ArXiv:** [2402.03804](https://arxiv.org/abs/2402.03804), [Local link](Papers%5C2402.03804v1_ReLU2%20Wins.pdf), Hash: 074b27ba8d6ebf98479d64d091f2ffec, *Added: 2025-01-17* 

### DISTILLM: Towards Streamlined Distillation for Large Language Models

*Jongwoo Ko, Sungnyun Kim, Tianyi Chen, Se-Young Yun*

**Summary:** We introduce DISTILLM, a more effective and efficient knowledge distillation framework for auto-regressive language models.

**ArXiv:** [2402.03898](https://arxiv.org/abs/2402.03898), [Local link](Papers%5C2402.03898v2_DISTILLM.pdf), Hash: 12db597685345092443110477b968cd2, *Added: 2025-01-17* 

### Grandmaster-Level Chess Without Search

*Anian Ruoss, Grégoire Delétang, Sourabh Medapati, Jordi Grau-Moya, Li Kevin Wenliang, Elliot Catt, John Reid, Tim Genewein*

**Summary:** This paper investigates the impact of training at scale for chess using a 270M parameter transformer model with supervised learning on a dataset of 10 million chess games, reaching a Lichess blitz Elo of 2895 against humans and solving challenging chess puzzles without explicit search algorithms.

**ArXiv:** [2402.04494](https://arxiv.org/abs/2402.04494), [Local link](Papers%5C2402.04494v1_Grandmaster-Level%20Chess%20Without%20Search.pdf), Hash: 65fc08a5f98bd21e621e8a7d622c7a52, *Added: 2025-01-17* 

### Training Large Language Models for Reasoning through Reverse Curriculum Reinforcement Learning

*Zhiheng Xi, Wenxiang Chen, Boyang Hong, Senjie Jin, Rui Zheng, Wei He, Yiwen Ding, Shichun Liu, Xin Guo, Junzhe Wang, Honglin Guo, Wei Shen, Xiaoran Fan, Yuhao Zhou, Shihan Dou, Xiao Wang, Xinbo Zhang, Peng Sun, Tao Gui, Qi Zhang, Xuanjing Huang*

**Summary:** In this paper, we propose R3: Learning Reasoning through Reverse Curriculum Reinforcement Learning (RL), a novel method that employs only outcome supervision to achieve the benefits of process supervision for large language models.

**ArXiv:** [2402.05808](https://arxiv.org/abs/2402.05808), [Local link](Papers%5C2402.05808v2.pdf), Hash: 42efd41e64ac26f1c0c57f9eb44fccd1, *Added: 2025-01-17* 

### Training Large Language Models for Reasoning through Reverse Curriculum Reinforcement Learning

*Zhiheng Xi, Wenxiang Chen, Boyang Hong, Senjie Jin, Rui Zheng, Wei He, Yiwen Ding, Shichun Liu, Xin Guo, Junzhe Wang, Honglin Guo, Wei Shen, Xiaoran Fan, Yuhao Zhou, Shihan Dou, Xiao Wang, Xinbo Zhang, Peng Sun, Tao Gui, Qi Zhang, Xuanjing Huang*

**Summary:** In this paper, we propose R3: Learning Reasoning through Reverse Curriculum Reinforcement Learning (RL), a novel method that employs only outcome supervision to achieve the benefits of process supervision for large language models.

**ArXiv:** [2402.05808](https://arxiv.org/abs/2402.05808), [Local link](Papers%5C2402.05808v2_Reverse%20Curriculum%20Reinforcement%20Learning.pdf), Hash: 42efd41e64ac26f1c0c57f9eb44fccd1, *Added: 2025-01-17* 

### Chain-of-Thought Reasoning without Prompting

*Xuezhi Wang, Denny Zhou*

**Summary:** This study investigates whether large language models can reason effectively without prompting by altering the decoding process to uncover chain-of-thought reasoning paths.

**ArXiv:** [2402.10200](https://arxiv.org/abs/2402.10200), [Local link](Papers%5C2402.10200v2_Chain-of-Thought%20Reasoning%20without%20Prompting.pdf), Hash: 5fe4440b5ccee481c078eda2a2cf0575, *Added: 2025-01-17* 

### Chain of Thought Empowers Transformers to Solve Inherently Serial Problems

*Zhiyuan Li, Hong Liu, Denny Zhou, Tengyu Ma*

**Summary:** Instructing the model to generate a sequence of intermediate steps, a.k.a., a chain of thought (CoT), is a highly effective method to improve the accuracy of large language models (LLMs) on arithmetics and symbolic reasoning tasks.

**ArXiv:** [2307.05562](https://arxiv.org/abs/2307.05562), [Local link](Papers%5C2402.12875v3_Chain%20of%20Thought%20Empowers%20Transformers.pdf), Hash: 2bfaabd7d8dea75c933e4b3eab2f66b0, *Added: 2025-01-17* 

### MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases

*Zechun Liu, Changsheng Zhao, Forrest Iandola, Chen Lai, Yuandong Tian, Igor Fedorov, Yunyang Xiong, Ernie Chang, Yangyang Shi, Raghuraman Krishnamoorthi, Liangzhen Lai, Vikas Chandra*

**Summary:** This paper addresses the growing need for efficient large language models (LLMs) on mobile devices, driven by increasing cloud costs and latency concerns. We focus on designing top-quality LLMs with fewer than a billion parameters, a practical choice for mobile deployment.

**ArXiv:** [2402.14905](https://arxiv.org/abs/2402.14905), [Local link](Papers%5C2402.14905v2_mobilellm.pdf), Hash: 0fe2ef51d464a5cae1876f2382379428, *Added: 2025-01-17* 

### Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking

*Eric Zelikman, Yijia Shao, Varuna Jayasiri, Nick Haber, Noah D. Goodman*

**Summary:** We present Quiet-STaR, a generalization of STaR in which LMs learn to generate rationales at each token to explain future text, improving their predictions.

**ArXiv:** [2403.09629](https://arxiv.org/abs/2403.09629), [Local link](Papers%5C2403.09629v2_Quiet-STaR.pdf), Hash: 8cdec7acbfc428496678eee2a098fd1a, *Added: 2025-01-17* 

### Reinforcement Learning from Reflective Feedback (RLRF): Aligning and Improving LLMs via Fine-Grained Self-Reflection

*Kyungjae Lee, Dasol Hwang, Sunghyun Park, Youngsoo Jang, Moontae Lee*

**Summary:** To overcome challenges in aligning large language models (LLMs) with human preferences, we propose a novel framework: Reinforcement Learning from Reflective Feedback (RLRF), which leverages fine-grained feedback based on detailed criteria to improve the core capabilities of LLMs.

**ArXiv:** [2403.14238](https://arxiv.org/abs/2403.14238), [Local link](Papers%5C2403.14238v1_Reinforcement%20Learning%20from%20Reflective%20Feedback.pdf), Hash: 638fd83a2afcd870acde9aebfc69fe34, *Added: 2025-01-17* 

### The Unreasonable Ineffectiveness of the Deeper Layers

*Andrey Gromov, Kushal Tirumala, Hassan Shapourian, Paolo Glorioso, Daniel A. Roberts*

**Summary:** We empirically study a simple layer-pruning strategy for popular families of open-weight pretrained LLMs, finding minimal degradation of performance on different question-answering benchmarks until after a large fraction (up to half) of the layers are removed.

**ArXiv:** [2403.17887](https://arxiv.org/abs/2403.17887), [Local link](Papers%5C2403.17887v1_layer_pruning.pdf), Hash: 652bf63701a5726bcffd722e555344b4, *Added: 2025-01-17* 

### QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs

*Saleh Ashkboos, Amirkeivan Mohtashami, Maximilian L. Croci, Bo Li, Pashmina Cameron, Martin Jaggi, Dan Alistarh, Torsten Hoefler, James Hensman*

**Summary:** We introduce QuaRot, a new Quantization scheme based on Rotations, which is able to quantize LLMs end-to-end, including all weights, activations, and KV cache in 4 bits.

**ArXiv:** [2404.00456](https://arxiv.org/abs/2404.00456), [Local link](Papers%5C2404.00456v2_QuaRot_Outlier-Free%204-Bit%20Inference%20in%20Rotated%20LLMs.pdf), Hash: 3732a052ae6b60bd43d04b26735507da, *Added: 2025-01-17* 

### Mixture-of-Depths: Dynamically Allocating Compute in Transformer-Based Language Models

*David Raposo, Sam Ritter, Blake Richards, Timothy Lillicrap, Peter Conway Humphreys, Adam Santoro*

**Summary:** This work demonstrates that transformers can learn to dynamically allocate FLOPs or compute to specific positions in a sequence, optimizing the allocation along the sequence for different layers across the model depth.

**ArXiv:** [2404.02258](https://arxiv.org/abs/2404.02258), [Local link](Papers%5C2404.02258.pdf), Hash: f90acf2f1085d7d99cd7c9f6aa908008, *Added: 2025-01-17* 

### Talaria: Interactively Optimizing Machine Learning Models for Efficient Inference

*Fred Hohman, Chaoqun Wang, Jinmook Lee, Jochen Görtler, Dominik Moritz, Jeffrey P. Bigham, Zhile Ren, Cecile Foret, Qi Shan, Xiaoyi Zhang*

**Summary:** On-device machine learning (ML) moves computation from the cloud to personal devices, protecting user privacy and enabling intelligent user experiences.

**ArXiv:** [2404.03085](https://arxiv.org/abs/2404.03085), [Local link](Papers%5C2404.03085v1_talaria.pdf), Hash: 8f4f4f0eda22a495f23c1337142e775b, *Added: 2025-01-17* 

### Training LLMs over Neurally Compressed Text

*Brian Lester, Jaehoon Lee, Alex Alemi, Jeffrey Pennington, Adam Roberts, Jascha Sohl-Dickstein, Noah Constant*

**Summary:** In this paper, we explore the idea of training large language models (LLMs) over highly compressed text using Equal-Info Windows, a novel compression technique that allows for effective learning and improved performance on perplexity and inference speed benchmarks.

**ArXiv:** [2404.03626](https://arxiv.org/abs/2404.03626), [Local link](Papers%5C2404.03626.pdf), Hash: 53529e681ab369bfee9e63d32619212c, *Added: 2025-01-17* 

### Physics of Language Models: Part 3.3 Knowledge Capacity Scaling Laws

*Zeyuan Allen-Zhu, Yuanzhi Li*

**Summary:** Scaling laws describe the relationship between the size of language models and their capabilities. Unlike prior studies that evaluate a model's capability via loss or benchmarks, we estimate the number of knowledge bits a model stores.

**ArXiv:** [2404.05405](https://arxiv.org/abs/2404.05405), [Local link](Papers%5C2404.05405v1_physics_of_llm.pdf), Hash: 682a50a5b5813643d0afc1a494abe7fc, *Added: 2025-01-17* 

### Grokked Transformers are Implicit Reasoners: A Mechanistic Journey to the Edge of Generalization

*Boshi Wang, Xiang Yue, Yu Su, Huan Sun*

**Summary:** We study whether transformers can learn to implicitly reason over parametric knowledge, a skill that even the most capable language models struggle with. Focusing on two representative reasoning types, composition and comparison, we consistently find that transformers can learn implicit reasoning, but only through grokking , i.e., extended training far beyond overfitting.

**ArXiv:** [2405.15071](https://arxiv.org/abs/2405.15071), [Local link](Papers%5C2405.15071v1_grokking.pdf), Hash: 259f8dbea45f5563ea5f2d9b166d44ec, *Added: 2025-01-17* 

### MoEUT: Mixture-of-Experts Universal Transformers

*Róbert Csordás, Kazuki Irie, Jürgen Schmidhuber, Christopher Potts, Christopher D. Manning*

**Summary:** Previous work on Universal Transformers (UTs) has demonstrated the importance of parameter sharing across layers.

**ArXiv:** [2405.16039](https://arxiv.org/abs/2405.16039), [Local link](Papers%5C2405.16039v2_ixture-of-Experts%20Universal%20Transformers.pdf), Hash: 1b4315d63eeb14a5f75fd1bcfc93dfe3, *Added: 2025-01-17* 

### Accessing GPT-4 Level Mathematical Olympiad Solutions via Monte Carlo Tree Self-refine with LLaMa-3 8B: A Technical Report

*Di Zhang, Xiaoshui Huang, Dongzhan Zhou, Yuqiang Li, Wanli Ouyang*

**Summary:** This paper introduces the MCT Self-Refine (MCTSr) algorithm, an innovative integration of Large Language Models (LLMs) with Monte Carlo Tree Search (MCTS), designed to enhance performance in complex mathematical reasoning tasks.

**ArXiv:** [2406.07394](https://arxiv.org/abs/2406.07394), [Local link](Papers%5C2406.07394v2_mctsr.pdf), Hash: a900fa60e9971a7181b21bae275e94dd, *Added: 2025-01-17* 

### What Matters in Transformers? Not All Attention Is Needed

*Shwai He, Guoheng Sun, Zhenyu Shen, Ang Li*

**Summary:** This work investigates redundancy across different modules within Transformers, including Blocks, MLP, and Attention layers, using a similarity-based metric. The authors found that a large portion of attention layers exhibit excessively high similarity and can be pruned without degrading performance.

**ArXiv:** [2406.15786](https://arxiv.org/abs/2406.15786), [Local link](Papers%5C2406.15786v6_Not%20All%20Attention%20is%20Needed.pdf), Hash: 605a6dc0714bc49a58e0810eb29c985c, *Added: 2025-01-17* 

### What Matters in Transformers?

*Shwai He, Guoheng Sun, Zhenyu Shen, Ang Li*

**Summary:** Despite the critical role of attention layers in distinguishing transformers from other architectures, a large portion of these layers exhibit excessively high similarity and can be pruned without degrading performance.

**ArXiv:** [2406.15786](https://arxiv.org/abs/2406.15786), [Local link](Papers%5C2406.15786v6_What%20Matters%20in%20Transformers.pdf), Hash: 605a6dc0714bc49a58e0810eb29c985c, *Added: 2025-01-17* 

### ARES: Alternating Reinforcement Learning and Supervised Fine-Tuning for Enhanced Multi-Modal Chain-of-Thought Reasoning Through Diverse AI Feedback

*Ju-Seung Byun, Jiyun Chun, Jihyung Kil, Andrew Perrault*

**Summary:** We propose a two-stage algorithm ARES that Alternates REinforcement Learning (RL) and Supervised Fine-Tuning (SFT). First, we request the Teacher to score how much each sentence contributes to solving the problem in a Chain-of-Thought (CoT). This sentence-level feedback allows us to consider individual valuable segments, providing more granular rewards for the RL procedure. Second, we ask the Teacher to correct the wrong reasoning after the RL stage.

**ArXiv:** [2407.00087](https://arxiv.org/abs/2407.00087), [Local link](Papers%5C2407.00087v1.pdf), Hash: c8145601b30ea1f059cb8894448c3b9e, *Added: 2025-01-17* 

### Q-Sparse: All Large Language Models can be Fully Sparsely-Activated

*Hongyu Wang, Shuming Ma, Ruiping Wang, Furu Wei*

**Summary:** We introduce, Q-Sparse , a simple yet effective approach to training sparsely-activated large language models (LLMs). Q-Sparse enables full sparsity of acti-vations in LLMs which can bring significant efficiency gains in inference.

**ArXiv:** [2407.10969](https://arxiv.org/abs/2407.10969), [Local link](Papers%5C2407.10969v1_Q-sparse.pdf), Hash: fd79b84fc95a6af58f65017ab49255fa, *Added: 2025-01-17* 

### Large Language Monkeys: Scaling Inference Compute with Repeated Sampling

*Bradley Brown, Jordan Juravsky, Ryan Ehrlich, Ronald Clark, Quoc V. Le, Christopher Ré, Azalia Mirhoseini*

**Summary:** Scaling the amount of compute used to train language models has dramatically improved their capabilities. However, when it comes to inference, we often limit the amount of compute to only one attempt per problem.

**ArXiv:** [2407.21787](https://arxiv.org/abs/2407.21787), [Local link](Papers%5C2407.21787v1_Scaling%20Inference%20Compute%20with%20repeated%20sampling.pdf), Hash: 54d27c34abf8b6b2893bc0646bc32740, *Added: 2025-01-17* 

### Scaling LLM Test-Time Compute Optimally Can Be More Effective than Scaling Model Parameters

*Charlie Snell, Jaehoon Lee, Kelvin Xu, Aviral Kumar*

**Summary:** In this paper, we study the scaling of inference-time computation in LLMs and find that a 'compute-optimal' scaling strategy can improve the efficiency of test-time compute scaling by more than 4× compared to a best-of-N baseline.

**ArXiv:** [2408.03314](https://arxiv.org/abs/2408.03314), [Local link](Papers%5C2408.03314v1_Scaling%20LLM%20Test-Time%20Compute%20Optimally.pdf), Hash: 90d86e47d5568dbce5599bc323c1140b, *Added: 2025-01-17* 

### Iteration of Thought: Leveraging Inner Dialogue for Autonomous Large Language Model Reasoning

*Santosh Kumar Radha, Yasamin Nouri Jelyani, Ara Ghukasyan, Oktay Goktas*

**Summary:** Motivated by the insight that iterative human engagement is a common and effective means of leveraging the advanced language processing power of large language models (LLMs), the Iteration of Thought (IoT) framework is proposed to enhance LLM responses by generating 'thought'-provoking prompts vis-a-vis input query and the current iteration of an LLM's response.

**ArXiv:** [2409.12618](https://arxiv.org/abs/2409.12618), [Local link](Papers%5C2409.12618v1_Iteration%20of%20Thought.pdf), Hash: 63851d16d472a122bc6875070a6615df, *Added: 2025-01-17* 

### Training Language Models to Self-Correct via Reinforcement Learning

*Aviral Kumar, Vincent Zhuang, Rishabh Agarwal, Yi Su, JD Co-Reyes, Avi Singh, Kate Baumli, Shariq Iqbal, Colton Bishop, Rebecca Roelofs, Lei M Zhang, Kay McKinney, Disha Shrivastava, Cosmin Paduraru, George Tucker, Doina Precup, Feryal Behbahani, Aleksandra Faust*

**Summary:** We develop a multi-turn online reinforcement learning (RL) approach, SCoRe, that significantly improves an LLM's self-correction ability using entirely self-generated data.

**ArXiv:** [2409.12917](https://arxiv.org/abs/2409.12917), [Local link](Papers%5C2409.12917v1_Training%20Language%20Models%20to%20Self-Correct%20via.pdf), Hash: 5d2d16a325b89d46187267d35e2ca0b0, *Added: 2025-01-17* 

### GSM-Symbolic: Understanding the Limitations of Mathematical Reasoning in Large Language Models

*Iman Mirzadeh, Keivan Alizadeh Hooman Shahrokhi, Oncel Tuzel, Samy Bengio, Mehrdad Farajtabar*

**Summary:** We introduce GSM-Symbolic, an improved benchmark created from symbolic templates that allow for the generation of a diverse set of questions to assess mathematical reasoning capabilities of models more reliably.

**ArXiv:** [2410.05229](https://arxiv.org/abs/2410.05229), [Local link](Papers%5C2410.05229v1_GSM-Symbolic.pdf), Hash: 0116c4a8ea5e9c013334193b2aed1ef9, *Added: 2025-01-17* 

### Value Residual Learning for Alleviating Attention Concentration in Transformers

*Zhanchao Zhou, Tianyi Wu, Zhiyun Jiang, Zhenzhong Lan*

**Summary:** Transformers can capture long-range dependencies using self-attention, allowing tokens to attend to all others directly. However, stacking multiple attention layers leads to attention concentration. One natural way to address this issue is to use cross-layer attention, allowing information from earlier layers to be directly accessible to later layers.

**ArXiv:** [2410.17897](https://arxiv.org/abs/2410.17897), [Local link](Papers%5C2410.17897v1_VALUE%20RESIDUAL%20LEARNING.pdf), Hash: e51bbd92587e0129b79cafa148e2fcec, *Added: 2025-01-17* 

### Scaling Laws for Precision

*Tanishq Kumar, Zachary Ankner, Benjamin F. Spector, Blake Bordelon, Niklas Muennighoff, Mansheej Paul, Cengiz Pehlevan, Christopher Ré, Aditi Raghunathan*

**Summary:** Low precision training and inference affect both the quality and cost of language models, but current scaling laws do not account for this. This work devises 'precision-aware' scaling laws for both training and inference.

**ArXiv:** [2411.04330](https://arxiv.org/abs/2411.04330), [Local link](Papers%5C2411.04330v1_Scaling%20Laws%20for%20Precision.pdf), Hash: 17fa88dae0abb80379c06931ff1a06f7, *Added: 2025-01-17* 

### Convolutional Differentiable Logic Gate Networks

*Felix Petersen, Hilde Kuehne, Christian Borgelt, Julian Welzel, Stefano Ermon*

**Summary:** With the increasing inference cost of machine learning models, there is a growing interest in models with fast and efficient inference. Recently, an approach for learning logic gate networks directly via a differentiable relaxation was proposed. Logic gate networks are faster than conventional neural network approaches because their inference only requires logic gate operators such as NAND, OR, and XOR, which are the underlying building blocks of current hardware and can be efficiently executed.

**ArXiv:** [2411.04732](https://arxiv.org/abs/2411.04732), [Local link](Papers%5C2411.04732v1_Convolutional%20Differentiable%20Logic%20Gate%20Networks.pdf), Hash: 9d34740e0103d1c3361cefdeca11e327, *Added: 2025-01-17* 

### BitNet a4.8: 4-bit Activations for 1-bit LLMs

*Hongyu Wang, Shuming Ma, Furu Wei*

**Summary:** We introduce BitNet a4.8, enabling 4-bit activations for 1-bit Large Language Models (LLMs). It employs a hybrid quantization and sparsification strategy to mitigate the quantization errors.

**ArXiv:** [2411.04965v1](https://arxiv.org/abs/2411.04965v1), [Local link](Papers%5C2411.04965v1_4-bit%20Activations%20for%201-bit%20LLMs.pdf), Hash: 6c0d93572f2ee1d63474ef440d702970, *Added: 2025-01-17* 

### A Hybrid-head Architecture for Small Language Models

*Xin Dong, Yonggan Fu, Shizhe Diao, Wonmin Byeon, Zijia Chen, Ameya Sunil Mahabaleshwarkar, Shih-Yang Liu, Matthijs Van Keirsbilck, Min-Hung Chen, Yoshi Suhara, Yingyan Celine Lin, Jan Kautz, Pavlo Molchanov*

**Summary:** We propose Hymba, a family of small language models featuring a hybrid-head parallel architecture that integrates transformer attention mechanisms with state space models (SSMs) for enhanced efficiency.

**ArXiv:** [2411.13676](https://arxiv.org/abs/2411.13676), [Local link](Papers%5C2411.13676v1_Hymba.pdf), Hash: 8151724137e6538fb2f0bd0bfbbdb399, *Added: 2025-01-17* 

### Flow Matching Guide and Code

*Yaron Lipman, Marton Havasi, Peter Holderrieth, Neta Shaul, Matt Le, Brian Karrer, Ricky T. Q. Chen, David Lopez-Paz, Heli Ben-Hamu, Itai Gat*

**Summary:** Flow Matching (FM) is a recent framework for generative modeling that has achieved state-of-the-art performance across various domains, including image, video, audio, speech, and biological structures. This guide offers a comprehensive and self-contained review of FM, covering its mathematical foun-dations, design choices, and extensions.

**ArXiv:** [2412.06264](https://arxiv.org/abs/2412.06264), [Local link](Papers%5C2412.06264v1_Flow%20Matching%20Guide%20and%20Code.pdf), Hash: 2cee47d8cecdf2cca0ce6741a15e5d6b, *Added: 2025-01-17* 

### Training Large Language Models to Reason in a Continuous Latent Space

*Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, Yuandong Tian*

**Summary:** We introduce Coconut (ChainofContinuousThought), a novel paradigm that utilizes the last hidden state of LLMs as continuous thought for reasoning, enabling advanced reasoning patterns and outperforming CoT in certain logical tasks.

**ArXiv:** [2412.06769](https://arxiv.org/abs/2412.06769), [Local link](Papers%5C2412.06769v1_coconut.pdf), Hash: 87493b967bec98bd1067df536c155310, *Added: 2025-01-17* 

### µNAS: Constrained Neural Architecture Search for Microcontrollers

*Edgar Liberis, Łukasz Dudziak, Nicholas D. Lane*

**Summary:** IoT devices are powered by microcontroller units (MCUs) which are extremely resource-scarce: a typical MCU may have an underpowered processor and around 64 KB of mem-ory and persistent storage.

**ArXiv:** [2010.11267](https://arxiv.org/abs/2010.11267), [Local link](Papers%5C3437984.3458836_%CE%BCNAS.pdf), Hash: 97de48ec413ce403e8bfb384cf2d79e1, *Added: 2025-01-17* 

### DeepSeek-V3 Technical Report

*DeepSeek-AI*

**Summary:** We present DeepSeek-V3, a strong Mixture-of-Experts (MoE) language model with 671B total parameters with 37B activated for each token.

**ArXiv:** [2108.07732](https://arxiv.org/abs/2108.07732), [Local link](Papers%5CDeepSeek_V3.pdf), Hash: 03697914c89904d28cc11c7c596901cc, *Added: 2025-01-17* 

### Memory Layers at Scale

*Vincent-Pierre Berges, Barlas Oğuz, Daniel Haziza, Wen-tau Yih, Luke Zettlemoyer, Gargi Gosh*

**Summary:** This work takes memory layers beyond proof-of-concept, proving their utility at contemporary scale. On downstream tasks, language models augmented with our improved memory layer outperform dense models with more than twice the computation budget, as well as mixture-of-expert models when matched for both compute and parameters.

**ArXiv:** N/A, [Local link](Papers%5CMemory%20Layers%20at%20Scale.pdf), Hash: 2a6e4743aa6dd2fab537b43830cf9964, *Added: 2025-01-17* 

### Optimal Brain Damage

*Yann Le Cun, John S. Denker, Sara A. Solla*

**Summary:** We have used information-theoretic ideas to derive a class of practical and nearly optimal schemes for adapting the size of a neural network by removing unimportant weights from a network, leading to better generalization, fewer training examples required, and improved speed of learning and/or classification.

**ArXiv:** N/A, [Local link](Papers%5CNIPS-1989-optimal-brain-damage-Paper.pdf), Hash: c1bfc00c11a88f7d9abbc1615f100613, *Added: 2025-01-17* 

