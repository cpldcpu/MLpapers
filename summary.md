#### 1. TRAINED TERNARY QUANTIZATION

*Chenzhuo Zhu, Song Han, Huizi Mao, William J. Dally*

**Summary:** Deep neural networks are widely used in machine learning applications. However, the deployment of large neural networks models can be difficult to deploy on mobile devices with limited power budgets. To solve this problem, we propose Trained Ternary Quantization (TTQ), a method that can reduce the precision of weights in neural networks to ternary values.

**ArXiv:** 1612.01064, [Link](Papers/1612.01064v3.pdf)
#### 2. Averaging Weights Leads to Wider Optima and Better Generalization

*Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov, Dmitry Vetrov, Andrew Gordon Wilson*

**Summary:** Deep neural networks are typically trained by optimizing a loss function with an SGD variant, in conjunction with a decaying learning rate, until convergence. We show that simple averaging of multiple points along the trajectory of SGD, with a cyclical or constant learning rate, leads to better generalization than conventional training.

**ArXiv:** 1803.05407, [Link](Papers/1803.05407v3_Stochastic_Weight_Averaging.pdf)
#### 3. World Models

*David Ha, Jürgen Schmidhuber*

**Summary:** We explore building generative neural network models of popular reinforcement learning environments.

**ArXiv:** 1803.10122, [Link](Papers/1803.10122.pdf)
#### 4. SqueezeNext: Hardware-Aware Neural Network Design

*Amir Gholami, Kiseok Kwon, Bichen Wu, Zizheng Tai, Xiangyu Yue, Peter Jin, Sicheng Zhao, Kurt Keutzer*

**Summary:** One of the main barriers for deploying neural networks on embedded systems has been large memory and power consumption of existing neural networks. In this work, we introduce SqueezeNext, a new family of neural network architectures whose design was guided by considering previous architectures such as SqueezeNet, as well as by simulation results on a neural network accelerator.

**ArXiv:** 1803.10615, [Link](Papers/1803.10615v2_squeezenext.pdf)
#### 5. Deep k-Means: Re-Training and Parameter Sharing with Harder Cluster Assignments for Compressing Deep Convolutions

*Junru Wu, Yue Wang, Zhenyu Wu, Zhangyang Wang, Ashok Veeraraghavan, Yingyan Lin*

**Summary:** We propose a scheme for compressing convolutions through k-means clustering on weights, achieving compression via weight-sharing. A spectrally relaxed k-means regularization is introduced to make harder assignments of convolutional layer weights during re-training.

**ArXiv:** 1803.06834, [Link](Papers/1806.09228v1_palettization.pdf)
#### 6. Improving Neural Network Quantization without Retraining using Outlier Channel Splitting

*Ritchie Zhao, Yuwei Hu, Jordan Dotzel, Christopher De Sa, Zhiru Zhang*

**Summary:** Quantization can improve the execution latency and energy efficiency of neural networks on both commodity GPUs and specialized accelerators.

**ArXiv:** 1901.09504, [Link](Papers/1901.09504v3_outlier channel splitting.pdf)
#### 7. LEARNED STEPSIZEQUANTIZATION

*Steven K. Esser, Jeffrey L. McKinstry, Deepika Bablani, Rathinakumar Appuswamy, Dharmendra S. Modha*

**Summary:** Here, we present a method for training deep networks with low precision operations at inference time, Learned Step Size Quantization, that achieves the highest accuracy to date on the ImageNet dataset when using models with weights and activations quantized to 2-, 3- or 4-bits of precision.

**ArXiv:** 1902.08153, [Link](Papers/1902.08153v3_good_paper_qat.pdf)
#### 8. All You Need is a Few Shifts: Designing Efficient Convolutional Neural Networks for Image Classification

*Weijie Chen, Di Xie, Yuan Zhang, Shiliang Pu*

**Summary:** Shift operation is an efficient alternative over depthwise separable convolution. However, it is still bottlenecked by its implementation manner, namely memory movement. To put this direction forward, a new and novel basic component named Sparse Shift Layer (SSL) is introduced in this paper to construct efficient convolutional neural networks.

**ArXiv:** 1903.05285, [Link](Papers/1903.05285.pdf)
#### 9. Character Region Awareness for Text Detection

*Youngmin Baek, Bado Lee, Dongyoon Han, Sangdoo Yun, Hwalsuk Lee*

**Summary:** In this paper, we propose a new scene text detection method to effectively detect text area by exploring each character and affinity between characters.

**ArXiv:** 1904.01941, [Link](Papers/1904.01941v1_craft_text_detection.pdf)
#### 10. Searching for MobileNetV3

*Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam*

**Summary:** We present the next generation of MobileNets based on a combination of complementary search techniques as well as a novel architecture design.

**ArXiv:** 1905.02244, [Link](Papers/1905.02244v5_mobilenetv3.pdf)
#### 11. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks

*Mingxing Tan, Quoc V. Le*

**Summary:** Convolutional Neural Networks (ConvNets) are commonly developed at a fixed resource budget, and then scaled up for better accuracy if more resources are available. In this paper, we systematically study model scaling and identify that carefully balancing network depth, width, and resolution can lead to better performance.

**ArXiv:** 1905.11946, [Link](Papers/1905.11946_efficientnet.pdf)
#### 12. DeepShift: Towards Multiplication-Less Neural Networks

*Mostafa Elhoushi, Zihao Chen, Farhan Shafig, Ye Henry Tian, Joey Yiwei Li*

**Summary:** The high computation, memory, and power budgets of inferring convolutional neural networks (CNNs) are major bottlenecks of model deployment to edge computing platforms, e.g., mobile devices and IoT. Moreover, training CNNs is time and energy-intensive even on high-grade servers.

**ArXiv:** 1905.13298, [Link](Papers/1905.13298_deepshift.pdf)
#### 13. ADDITIVE POWERS-OF-TWO QUANTIZATION: AN EFFICIENT NON-UNIFORM DISCRETIZATION FOR NEURAL NETWORKS

*Yuhang Li, Xin Dong, Wei Wang*

**Summary:** We propose Additive Powers-of-Two (APoT) quantization, an efficient non-uniform quantization scheme for the bell-shaped and long-tailed distribution of weights and activations in neural networks.

**ArXiv:** 1909.13144, [Link](Papers/1909.13144v2_expcoding.pdf)
#### 14. AdderNet: Do We Really Need Multiplications in Deep Learning?

*Hanting Chen, Yunhe Wang, Chunjing Xu, Boxin Shi, Chao Xu, Qi Tian, Chang Xu*

**Summary:** Adder networks (AdderNets) are presented to replace massive multiplications in deep neural networks with cheaper additions, reducing computation costs. The proposed AdderNets can achieve high accuracy on the ImageNet dataset without any multiplication in convolution layer.

**ArXiv:** 1912.13200, [Link](Papers/1912.13200.pdf)
#### 15. AdderSR: Towards Energy Efﬁcient Image Super-Resolution

*Dehua Song, Yunhe Wang, Hanting Chen, Chang Xu, Chunjing Xu, Dacheng Tao*

**Summary:** This paper studies the single image super-resolution problem using adder neural networks (AdderNets).

**ArXiv:** 2009.08891, [Link](Papers/2009.08891.pdf)
#### 16. ANIMAGE IS WORTH 16X16 W ORDS : TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE

*Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby*

**Summary:** We show that a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks, and Vision Transformer (ViT) attains excellent results compared to state-of-the-art convolutional networks while requiring substantially fewer computational resources to train.

**ArXiv:** 2010.11929, [Link](Papers/2010.11929.pdf)
#### 17. Sparsity in Deep Learning: Pruning and growth for efficient inference and training in neural networks

*TORSTEN HOEFLER, ETH Zürich, Switzerland, DAN ALISTARH, IST Austria, Austria, TAL BEN-NUN, ETH Zürich, Switzerland, NIKOLI DRYDEN, ETH Zürich, Switzerland, ALEXANDRA PESTE, IST Austria, Austria*

**Summary:** The growing energy and performance costs of deep learning have driven the community to reduce the size of neural networks by selectively pruning components. Similarly to their biological counterparts, sparse networks generalize just as well, if not better than, the original dense networks.

**ArXiv:** 2102.00554, [Link](Papers/2102.00554v1_sparsity.pdf)
#### 18. VS-Q UANT: PER-VECTOR SCALED QUANTIZATION FOR ACCURATE LOW-PRECISION NEURAL NETWORK INFERENCE

*Steve Dai, Rangharajan Venkatesan, Haoxing Ren, Brian Zimmer, William J. Dally, Brucek Khailany*

**Summary:** Quantization enables efficient acceleration of deep neural networks by reducing model memory footprint and exploiting low-cost integer math hardware units.

**ArXiv:** 2102.04503, [Link](Papers/2102.04503v1_per_vector_quant.pdf)
#### 19. Network Quantization with Element-wise Gradient Scaling

*Junghyup Lee, Dohyung Kim, Bumsub Ham*

**Summary:** We propose an element-wise gradient scaling (EWGS), a simple yet effective alternative to the straight-through estimator, training a quantized network better than the STE in terms of stability and accuracy.

**ArXiv:** 2104.00903, [Link](Papers/2104.00903v1_STE_Optimization_EWGS.pdf)
#### 20. LeViT: a Vision Transformer in ConvNet’s Clothing for Faster Inference

*Benjamin Graham, Alaaeldin El-Nouby, Hugo Touvron, Pierre Stock, Armand Joulin, Hervé Jégou, Matthijs Douze*

**Summary:** We design a family of image classification architectures that optimize the trade-off between accuracy and efficiency in a high-speed regime.

**ArXiv:** 2104.01136, [Link](Papers/2104.01136v2_levit_light_weight_vision.pdf)
#### 21. DKM: DIFFERENTIABLE k-MEANS CLUSTERING LAYER FOR NEURAL NETWORK COMPRESSION

*Minsik Cho, Keivan Alizadeh-Vahid, Saurabh Adya, Mohammad Rastegari*

**Summary:** We propose a differentiable k-means clustering layer (DKM) and its application to train-time weight-clustering for DNN model compression. DKM delivers superior compression and accuracy trade-off on ImageNet1k and GLUE benchmarks.

**ArXiv:** 2108.12659v4, [Link](Papers/2108.12659v4_palettization.pdf)
#### 22. Primer: Searching for Efficient Transformers for Language Modeling

*David R. So, Wojciech Mańke, Hanxiao Liu, Zihang Dai, Noam Shazeer, Quoc V. Le*

**Summary:** Large Transformer models have been central to recent advances in natural language processing. The training and inference costs of these models, however, have grown rapidly and become prohibitively expensive.

**ArXiv:** 2109.08668, [Link](Papers/2109.08668v2_primer_llm_architecture_search.pdf)
#### 23. Rare Gems: Finding Lottery Tickets at Initialization

*Kartik Sreenivasan, Jy-yong Sohn, Liu Yang, Matthew Grindle, Alliot Nagele, Hongyi Wang, Eric Xing, Kangwook Lee, Dimitris Papailiopoulos*

**Summary:** In this work, we resolve the open problem of finding lottery tickets at initialization that beat current baselines by proposing GEM-MINER.

**ArXiv:** 2202.12002, [Link](Papers/2202.12002v2_Finding Lottery Tickets at Initialization.pdf)
#### 24. Optimal Clipping and Magnitude-aware Differentiation for Improved Quantization-aware Training

*Charbel Sakr, Steve Dai, Rangharajan Venkatesan, Brian Zimmer, William J. Dally, Brucek Khailany*

**Summary:** This paper proposes Optimally Clipped Tensors And Vectors (OCTA V), a recursive algorithm to determine MSE-optimal clipping scalars for reducing quantization noise and improving the accuracy of quantization-aware training (QAT). It also introduces magnitude-aware differentiation to further enhance accuracy.

**ArXiv:** 2206.06501, [Link](Papers/2206.06501v1_optimal_clipping.pdf)
#### 25. LLM.int8() : 8-bit Matrix Multiplication for Transformers at Scale

*Tim Dettmers, Mike Lewis, Younes Belkada, Luke Zettlemoyer*

**Summary:** Large language models have been widely adopted but require significant GPU memory for inference. We develop a procedure for Int8 matrix multiplication for feed-forward and attention projection layers in transformers, which cut the memory needed for inference by half while retaining full precision performance.

**ArXiv:** 2208.07339, [Link](Papers/2208.07339.pdf)
#### 26. GPTQ: A CCURATE POST-TRAINING QUANTIZATION FOR GENERATIVE PRE-TRAINED TRANSFORMERS

*Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh*

**Summary:** GPTQ proposes a one-shot weight quantization method based on approximate second-order information that accurately and efficiently compresses GPT models, allowing execution of an 175 billion-parameter model inside a single GPU for generative inference.

**ArXiv:** 2210.17323, [Link](Papers/2210.17323.pdf)
#### 27. GPTQ: A CURATE POST-TRAINING QUANTIZATION FOR GENERATIVE PRE-TRAINED TRANSFORMERS

*Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh*

**Summary:** GPTQ introduces a one-shot weight quantization method based on approximate second-order information, which can accurately and efficiently compress GPT models with minimal accuracy loss.

**ArXiv:** 2210.17323, [Link](Papers/2210.17323v2_GPTQ.pdf)
#### 28. Symbolic Discovery of Optimization Algorithms

*Xiangning Chen, Chen Liang, Da Huang, Esteban Real, Kaiyuan Wang, Yao Liu, Hieu Pham, Xuanyi Dong, Thang Luong, Cho-Jui Hsieh, Yifeng Lu, Quoc V. Le*

**Summary:** We present a method to formulate algorithm discovery as program search, and apply it to discover optimization algorithms for deep neural network training.

**ArXiv:** 2302.06675, [Link](Papers/2302.06675v4_Lion_Opitmizer.pdf)
#### 29. Multiplication-Free Transformer Training via Piecewise Affine Operations

*Atli Kosson, Martin Jaggi*

**Summary:** We replace multiplication with a cheap piecewise affine approximation in transformers, making them fully and jointly piecewise affine in both inputs and weights without changes to the training hyperparameters.

**ArXiv:** 2305.17190, [Link](Papers/2305.17190v2_mulfree.pdf)
#### 30. A WQ: A ctivation-aware W eight Q uantization for LLM Compression and Acceleration

*Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Xingyu Dang, Chuang Gan, Song Han*

**Summary:** In this paper, we propose Activation-aware Weight Quantization (AWQ), a hardware-friendly approach for Large Language Model low-bit weight-only quantization by searching for the optimal per-channel scaling that protects salient weights by observing the activation.

**ArXiv:** 2306.00978, [Link](Papers/2306.00978.pdf)
#### 31. Patch n' Pack: NaViT, a Vision Transformer for any Aspect Ratio and Resolution

*Mostafa Dehghani, Basil Mustafa, Josip Djolonga, Jonathan Heek, Matthias Minderer, Mathilde Caron, Andreas Steiner, Joan Puigcerver, Robert Geirhos, Ibrahim Alabdulmohsin, Avital Oliver Piotr Padlewski, Alexey Gritsenko, Mario Lučić, Neil Houlsby*

**Summary:** We propose NaViT(Native Resolution ViT) which uses sequence packing during training to process inputs of arbitrary resolutions and aspect ratios, improving training efficiency for large-scale supervised and contrastive image-text pretraining.

**ArXiv:** 2306.14773, [Link](Papers/2307.06304.pdf)
#### 32. Explaining grokking through circuit efficiency

*Vikrant Varma, Rohin Shah, Zachary Kenton, János Kramár, Ramana Kumar*

**Summary:** One of the most surprising puzzles in neural network generalisation is grokking: a network with perfect training accuracy but poor generalisation will, upon further training, transition to perfect generalisation.

**ArXiv:** 2309.02390, [Link](Papers/2309.02390v1_grokking.pdf)
#### 33. BitNet: Scaling 1-bit Transformers for Large Language Models

*Hongyu Wang, Shuming Ma, Li Dong, Shaohan Huang, Huaijie Wang, Lingxiao Ma, Fan Yang, Ruiping Wang, Yi Wu, Furu Wei*

**Summary:** We introduce BitNet, a scalable and stable 1-bit Transformer architecture designed for large language models. It reduces memory footprint and energy consumption while maintaining competitive performance compared to state-of-the-art quantization methods and FP16 Transformer baselines.

**ArXiv:** 2310.11453, [Link](Papers/2310.11453_bitnet.pdf)
#### 34. FP6-LLM: Efficiently Serving Large Language Models Through FP6-Centric Algorithm-System Co-Design

*Haojun Xia, Zhen Zheng, Xiaoxia Wu, Shiyang Chen, Zhewei Yao, Stephen Youn, Arash Bakhtiari, Michael Wyatt, Donglin Zhuang, Zhongzhu Zhou, Olatunji Ruwase, Yuxiong He, Shuaiwen Leon Song*

**Summary:** Six-bit quantization (FP6) can effectively reduce the size of large language models (LLMs) and preserve the model quality consistently across varied applications.

**ArXiv:** 2401.14112, [Link](Papers/2401.14112v2_fp6_llm.pdf)
#### 35. No Free Prune: Information-Theoretic Barriers to Pruning at Initialization

*Tanishq Kumar, Kevin Luo, Mark Sellke*

**Summary:** The authors provide a theoretical explanation for the difficulties in finding sparse subnetworks without training the dense model, based on the concept of effective parameter count and mutual information between sparsity mask and data. They show that pruning near initialization may be infeasible and explain why lottery tickets exist but cannot be found fast.

**ArXiv:** 2402.01089, [Link](Papers/2402.01089v1_No Free Prune.pdf)
#### 36. MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases

*Zechun Liu, Changsheng Zhao, Forrest Iandola, Chen Lai, Yuandong Tian, Igor Fedorov, Yunyang Xiong, Ernie Chang, Yangyang Shi, Raghuraman Krishnamoorthi, Liangzhen Lai, Vikas Chandra*

**Summary:** This paper addresses the growing need for efficient large language models (LLMs) on mobile devices, driven by increasing cloud costs and latency concerns. We focus on designing top-quality LLMs with fewer than a billion parameters, a practical choice for mobile deployment.

**ArXiv:** 2402.14905, [Link](Papers/2402.14905v2_mobilellm.pdf)
#### 37. The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits

*Shuming Ma, Hongyu Wang, Lingxiao Ma, Lei Wang, Wenhui Wang, Shaohan Huang, Li Dong, Ruiping Wang, Jilong Xue, Furu Wei*

**Summary:** Recent research, such as BitNet [ WMD+23], is paving the way for a new era of 1- bit Large Language Models (LLMs). In this work, we introduce a 1-bit LLM variant, namely BitNet b1.58 , in which every single parameter (or weight) of the LLM is ternary {-1, 0, 1}. It matches the full-precision (i.e., FP16 or BF16) Transformer LLM with the same model size and training tokens in terms of both perplexity and end-task performance, while being significantly more cost-effective in terms of latency, memory, throughput, and energy consumption.

**ArXiv:** 2402.17764, [Link](Papers/2402.17764_1_58bit.pdf)
#### 38. The Unreasonable Ineffectiveness of the Deeper Layers

*Andrey Gromov, Kushal Tirumala, Hassan Shapourian, Paolo Glorioso, Daniel A. Roberts*

**Summary:** We empirically study a simple layer-pruning strategy for popular families of open-weight pretrained LLMs, finding minimal degradation of performance on different question-answering benchmarks until after a large fraction (up to half) of the layers are removed.

**ArXiv:** 2403.17887, [Link](Papers/2403.17887v1_layer_pruning.pdf)
#### 39. Mixture-of-Depths: Dynamically allocating compute in transformer-based language models

*David Raposo, Sam Ritter, Blake Richards, Timothy Lillicrap, Peter Conway Humphreys, Adam Santoro*

**Summary:** In this work, we demonstrate that transformers can learn to dynamically allocate FLOPs (or compute) to specific positions in a sequence, optimizing the allocation along the sequence for different layers across the model depth.

**ArXiv:** 2404.02258, [Link](Papers/2404.02258.pdf)
#### 40. Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction

*Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, Liwei Wang*

**Summary:** We present Visual AutoRegressive modeling (V AR), a new generation paradigm that redefines the autoregressive learning on images as coarse-to-fine "next-scale prediction" or "next-resolution prediction", diverging from the standard raster-scan "next-token prediction".

**ArXiv:** 2404.02905, [Link](Papers/2404.02905.pdf)
#### 41. Talaria : Interactively Optimizing Machine Learning Models for Efficient Inference

*Fred Hohman, Chaoqun Wang, Jinmook Lee, Jochen Görtler, Dominik Moritz, Jeffrey P. Bigham, Zhile Ren, Cecile Foret, Qi Shan, Xiaoyi Zhang*

**Summary:** On-device machine learning (ML) moves computation from the cloud to personal devices, protecting user privacy and enabling in-telligent user experiences. However, fitting models on devices with limited computational resources is challenging.

**ArXiv:** 2404.03085, [Link](Papers/2404.03085v1_talaria.pdf)
#### 42. Training LLMs over Neurally Compressed Text

*Brian Lester, Jaehoon Lee, Alex Alemi, Jeffrey Pennington, Adam Rutherford, Jascha Sohl-Dickstein, Noah Constant*

**Summary:** In this paper, we explore the idea of training large language models (LLMs) over highly compressed text using a novel compression technique called Equal-Info Windows.

**ArXiv:** 2308.07809, [Link](Papers/2404.03626.pdf)
#### 43. Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws

*Zeyuan Allen-Zhu, Yuanzhi Li*

**Summary:** Scaling laws describe the relationship between the size of language models and their capabilities. Unlike prior studies that evaluate a model's capability via loss or benchmarks, we estimate the number of knowledge bits a model stores.

**ArXiv:** 2404.05405, [Link](Papers/2404.05405v1_llm_scaling_meta.pdf)
#### 44. Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws

*Zeyuan Allen-Zhu, Yuanzhi Li*

**Summary:** Scaling laws describe the relationship between the size of language models and their capabilities. Unlike prior studies that evaluate a model's capability via loss or benchmarks, we estimate the number of knowledge bits a model stores.

**ArXiv:** 2404.05405, [Link](Papers/2404.05405v1_physics_of_llm.pdf)
#### 45. Grokked Transformers are Implicit Reasoners: A Mechanistic Journey to the Edge of Generalization

*Boshi Wang, Xiang Yue, Yu Su, Huan Sun*

**Summary:** We study whether transformers can learn to implicitly reason over parametric knowledge, a skill that even the most capable language models struggle with. Focusing on two representative reasoning types, composition and comparison, we consistently find that transformers can learn implicit reasoning, but only through grokking , i.e., extended training far beyond overfitting.

**ArXiv:** 2405.15071, [Link](Papers/2405.15071v1_grokking.pdf)
#### 46. Scalable MatMul-free Language Modeling

*Rui-Jie Zhu, Yu Zhang, Ethan Sifferman, Tyler Sheaves, Yiqiao Wang, Dustin Richmond, Peng Zhou, Jason K. Eshraghian*

**Summary:** This work shows that Matrix multiplication (MatMul) operations can be completely eliminated from large language models (LLMs) while maintaining strong performance at billion-parameter scales.

**ArXiv:** 2406.02528, [Link](Papers/2406.02528v1_matmul_free.pdf)
#### 47. Q-Sparse: All Large Language Models can be Fully Sparsely-Activated

*Hongyu Wang, Shuming Ma, Ruiping Wang, Furu Wei*

**Summary:** We introduce, Q-Sparse , a simple yet effective approach to training sparsely-activated large language models (LLMs). Q-Sparse enables full sparsity of acti-vations in LLMs which can bring significant efficiency gains in inference.

**ArXiv:** 2407.10969, [Link](Papers/2407.10969v1_Q-sparse.pdf)
#### 48. Least Squares Quantization in PCM

*STUART P. LLOYD*

**Summary:** Necessary conditions are found that the quanta and associated quantization intervals of an optimum finite quantization scheme must satisfy, with a given ensemble of signals to handle, the quantum values should be spaced more closely in the voltage regions where the signal amplitude is more likely to fall.

**ArXiv:** N/A, [Link](Papers/lloyd1982_least_squares_quant.pdf)
#### 49. ShiftAddNet: A Hardware-Inspired Deep Network

*Haoran You, Xiaohan Chen, Yongan Zhang, Chaojian Li, Sicheng Li, Zihao Liu, Zhangyang Wang, Yingyan Lin*

**Summary:** This paper introduces ShiftAddNet, a deep network inspired by energy-efficient hardware implementation that uses bit-shift and additive weight layers to reduce resource costs without compromising expressive capacity.

**ArXiv:** 1905.13298, [Link](Papers/NeurIPS-2020-shiftaddnet-a-hardware-inspired-deep-network-Paper.pdf)
#### 50. Optimal Brain Damage

*Yann Le Cun, John S. Denker, Sara A. Solla*

**Summary:** We have used information-theoretic ideas to derive a class of practical and nearly optimal schemes for adapting the size of a neural network.

**ArXiv:** N/A, [Link](Papers/NIPS-1989-optimal-brain-damage-Paper.pdf)
#### 51. OCP Micro scaling Format s (MX) Specification

*Bita Darvish Rouhani, Nitin Garegrat, Tom Savell , Ankit More, Kyung -Nam Han, Ritchie Zhao, Mathew Hall, Jasmine Klar, Eric Chung, Yuan Yu, Microsoft, Michael Schulte, Ralph Wittig, AMD, Ian Bratt, Nigel Stephens, Jelena Milanovic, John Brothers, Arm, Pradeep Dubey, Marius Cornea, Alexander Heinecke, Andres Rodriguez, Martin Langhammer, Intel, Summer Deng, Maxim Naumov, Meta, Paulius Micikevicius, Michael Siu, NVIDI A , Colin Verrilli, Qualcomm*

**Summary:** This Specification defines microscaling (MX) formats that are compliant with the Open Compute Project. It covers definitions, overview of MX-compliant formats, element data types, scale data types, basic operations, and references.

**ArXiv:** N/A, [Link](Papers/OCP_Microscaling Formats (MX) v1.0 Spec_Final.pdf)
