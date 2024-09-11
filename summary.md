#### 1. TRAINED TERNARY QUANTIZATION

*Chenzhuo Zhu, Song Han, Huizi Mao, William J. Dally*

**Summary:** Deep neural networks are widely used in machine learning applications. However, the deployment of large neural networks models can be difficult to deploy on mobile devices with limited power budgets. To solve this problem, we propose Trained Ternary Quantization (TTQ), a method that can reduce the precision of weights in neural networks to ternary values.

**ArXiv:** [1612.01064](https://arxiv.org/abs/1612.01064),[Local link](Papers/1612.01064v3.pdf)

#### 2. Averaging Weights Leads to Wider Optima and Better Generalization

*Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov, Dmitry Vetrov, Andrew Gordon Wilson*

**Summary:** We show that simple averaging of multiple points along the trajectory of SGD, with a cyclical or constant learning rate, leads to better generalization than conventional training.

**ArXiv:** [1803.05407](https://arxiv.org/abs/1803.05407),[Local link](Papers/1803.05407v3_Stochastic_Weight_Averaging.pdf)

#### 3. World Models

*David Ha, Jürgen Schmidhuber*

**Summary:** We explore building generative neural network models of popular reinforcement learning environments.

**ArXiv:** [1803.10122](https://arxiv.org/abs/1803.10122),[Local link](Papers/1803.10122.pdf)

#### 4. SqueezeNext: Hardware-Aware Neural Network Design

*Amir Gholami, Kiseok Kwon, Bichen Wu, Zizheng Tai, Xiangyu Yue, Peter Jin, Sicheng Zhao, Kurt Keutzer*

**Summary:** We introduce SqueezeNext, a new family of neural network architectures that match AlexNet's accuracy on the ImageNet benchmark with fewer parameters and achieve VGG-19 accuracy with significantly less parameters. These models can be used for speed-accuracy tradeoffs based on available resources.

**ArXiv:** [1803.10615](https://arxiv.org/abs/1803.10615),[Local link](Papers/1803.10615v2_squeezenext.pdf)

#### 5. Deep k-Means: Re-Training and Parameter Sharing with Harder Cluster Assignments for Compressing Deep Convolutions

*Junru Wu, Yue Wang, Zhenyu Wu, Zhangyang Wang, Ashok Veeraraghavan, Yingyan Lin*

**Summary:** We proposed a simple scheme for compressing convolutions through k-means clustering on the weights, achieving compression via weight-sharing by only recording K cluster centers and weight assignment indexes. We introduced spectrally relaxed k-means regularization to make hard assignments of convolutional layer weights during re-training.

**ArXiv:** [1806.09228](https://arxiv.org/abs/1806.09228),[Local link](Papers/1806.09228v1_palettization.pdf)

#### 6. Improving Neural Network Quantization without Retraining using Outlier Channel Splitting

*Ritchie Zhao, Yuwei Hu, Jordan Dotzel, Christopher De Sa, Zhiru Zhang*

**Summary:** Quantization can improve the execution latency and energy efficiency of neural networks on both commodity GPUs and specialized accelerators. The majority of existing literature focuses on training quantized DNNs, while this work examines the less-studied topic of quantizing a floating-point model without (re)training.

**ArXiv:** [1901.09504](https://arxiv.org/abs/1901.09504),[Local link](Papers/1901.09504v3_outlier channel splitting.pdf)

#### 7. Learned Step Size Quantization

*Steven K. Esser, Jeffrey L. McKinstry, Deepika Bablani, Rathinakumar Appuswamy, Dharmendra S. Modha*

**Summary:** Learned Step Size Quantization presents a method for training deep networks with low precision operations at inference time, achieving high accuracy on the ImageNet dataset using models quantized to 2-, 3- or 4-bits of precision.

**ArXiv:** [1902.08153](https://arxiv.org/abs/1902.08153),[Local link](Papers/1902.08153v3_good_paper_qat.pdf)

#### 8. All You Need is a Few Shifts: Designing Efficient Convolutional Neural Networks for Image Classification

*Weijie Chen, Di Xie, Yuan Zhang, Shiliang Pu*

**Summary:** Shift operation is an efficient alternative over depthwise separable convolution. However, it is still bottlenecked by its implementation manner, namely memory movement. To put this direction forward, a new and novel basic component named Sparse Shift Layer (SSL) is introduced in this paper to construct efficient convolutional neural networks.

**ArXiv:** [1903.05285](https://arxiv.org/abs/1903.05285),[Local link](Papers/1903.05285.pdf)

#### 9. Character Region Awareness for Text Detection

*Youngmin Baek, Bado Lee, Dongyoon Han, Sangdoo Yun, Hwalsuk Lee*

**Summary:** This paper proposes a new scene text detection method that effectively detects text areas by exploring each character and affinity between characters, significantly outperforming state-of-the-art detectors on benchmarks like TotalText and CTW-1500.

**ArXiv:** [1904.01941](https://arxiv.org/abs/1904.01941),[Local link](Papers/1904.01941v1_craft_text_detection.pdf)

#### 10. Searching for MobilenetV3

*Andrew Howard, Mark Sandler, Grace Chu, Liang-Chieh Chen, Bo Chen, Mingxing Tan, Weijun Wang, Yukun Zhu, Ruoming Pang, Vijay Vasudevan, Quoc V. Le, Hartwig Adam*

**Summary:** We present the next generation of MobileNets based on a combination of complementary search techniques as well as a novel architecture design.

**ArXiv:** [1905.02244](https://arxiv.org/abs/1905.02244),[Local link](Papers/1905.02244v5_mobilenetv3.pdf)

#### 11. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks

*Mingxing Tan, Quoc V. Le*

**Summary:** Convolutional Neural Networks (ConvNets) are commonly developed at a fixed resource budget, and then scaled up for better accuracy if more resources are available. In this paper, we systematically study model scaling and identify that carefully balancing network depth, width, and resolution can lead to better performance.

**ArXiv:** [1905.11946](https://arxiv.org/abs/1905.11946),[Local link](Papers/1905.11946_efficientnet.pdf)

#### 12. DeepShift: Towards Multiplication-Less Neural Networks

*Mostafa Elhoushi, Zihao Chen, Farhan Shafig, Ye Henry Tian, Joey Yiwei Li*

**Summary:** The high computation, memory, and power budgets of inferring convolutional neural networks (CNNs) are major bottlenecks of model deployment to edge computing platforms, e.g., mobile devices and IoT.

**ArXiv:** [1905.13298](https://arxiv.org/abs/1905.13298),[Local link](Papers/1905.13298_deepshift.pdf)

#### 13. ADDITIVE POWERS -OF-TWOQUANTIZATION : ANEFFICIENT NON-UNIFORM DISCRETIZATION FOR NEURAL NETWORKS

*Yuhang Li, Xin Dong, Wei Wang*

**Summary:** We propose Additive Powers-of-Two (APoT) quantization, an efficient non-uniform quantization scheme for the bell-shaped and long-tailed distribution of weights and activations in neural networks.

**ArXiv:** [1909.13144](https://arxiv.org/abs/1909.13144),[Local link](Papers/1909.13144v2_expcoding.pdf)

#### 14. AdderNet: Do We Really Need Multiplications in Deep Learning?

*Hanting Chen, Yunhe Wang, Chunjing Xu, Boxin Shi, Chao Xu, Qi Tian, Chang Xu*

**Summary:** Adder networks (AdderNets) are proposed to trade massive multiplications in deep neural networks, especially convolutional neural networks (CNNs), for much cheaper additions to reduce computation costs.

**ArXiv:** [1912.13200](https://arxiv.org/abs/1912.13200),[Local link](Papers/1912.13200.pdf)

#### 15. AdderSR: Towards Energy Efficient Image Super-Resolution

*Dehua Song, Yunhe Wang, Hanting Chen, Chang Xu, Chunjing Xu, Dacheng Tao*

**Summary:** This paper studies the single image super-resolution problem using adder neural networks (AdderNets) and proposes AdderSR to achieve comparable performance and visual quality to CNN baselines with a significant reduction in energy consumption.

**ArXiv:** [2009.08891](https://arxiv.org/abs/2009.08891),[Local link](Papers/2009.08891.pdf)

#### 16. ANIMAGE IS WORTH 16X16 W ORDS : TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE

*Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby*

**Summary:** We show that pure transformer applied directly to sequences of image patches can perform very well on image classification tasks. When pre-trained on large amounts of data and transferred to multiple mid-sized or small image recognition benchmarks, Vision Transformer (ViT) attains excellent results compared to state-of-the-art convolutional networks while requiring substantially fewer computational resources to train.

**ArXiv:** [2010.11929](https://arxiv.org/abs/2010.11929),[Local link](Papers/2010.11929.pdf)

#### 17. Sparsity in Deep Learning: Pruning and growth for efficient inference and training in neural networks

*TORSTEN HOEFLER, DAN ALISTARH, TAL BEN-NUN, NIKOLI DRYDEN, ALEXANDRA PESTE*

**Summary:** The growing energy and performance costs of deep learning have driven the community to reduce the size of neural networks by selectively pruning components.

**ArXiv:** [2102.00554](https://arxiv.org/abs/2102.00554),[Local link](Papers/2102.00554v1_sparsity.pdf)

#### 18. VS-Q UANT : PER-VECTOR SCALED QUANTIZATION FOR ACCURATE LOW-PRECISION NEURAL NETWORK INFERENCE

*Steve Dai, Rangharajan Venkatesan, Haoxing Ren, Brian Zimmer, William J. Dally, Brucek Khailany*

**Summary:** Quantization enables efficient acceleration of deep neural networks by reducing model memory footprint and exploiting low-cost integer math hardware units.

**ArXiv:** [2102.04503](https://arxiv.org/abs/2102.04503),[Local link](Papers/2102.04503v1_per_vector_quant.pdf)

#### 19. Network Quantization with Element-wise Gradient Scaling

*Junghyup Lee, Dohyung Kim, Bumsub Ham*

**Summary:** Network quantization aims at reducing bit-widths of weights and/or activations, particularly important for implementing deep neural networks with limited hardware resources.

**ArXiv:** [2104.00903](https://arxiv.org/abs/2104.00903),[Local link](Papers/2104.00903v1_STE_Optimization_EWGS.pdf)

#### 20. LeViT: a Vision Transformer in ConvNet’s Clothing for Faster Inference

*Benjamin Graham, Alaaeldin El-Nouby, Hugo Touvron, Pierre Stock, Armand Joulin, Hervé Jégou, Matthijs Douze*

**Summary:** We design a family of image classification architectures that optimize the trade-off between accuracy and efficiency in a high-speed regime.

**ArXiv:** [2104.01136](https://arxiv.org/abs/2104.01136),[Local link](Papers/2104.01136v2_levit_light_weight_vision.pdf)

#### 21. DKM: Differentiable k-Means Clustering Layer for Neural Network Compression

*Minsik Cho, Keivan Alizadeh-Vahid, Saurabh Adya, Mohammad Rastegari*

**Summary:** We propose a novel differentiable k-means clustering layer (DKM) and its application to train-time weight-clustering for DNN model compression.

**ArXiv:** [2108.12659v4](https://arxiv.org/abs/2108.12659v4),[Local link](Papers/2108.12659v4_palettization.pdf)

#### 22. Primer: Searching for Efficient Transformers for Language Modeling

*David R. So, Wojciech Mańke, Hanxiao Liu, Zihang Dai, Noam Shazeer, Quoc V. Le*

**Summary:** Large Transformer models have been central to recent advances in natural language processing. The training and inference costs of these models, however, have grown rapidly and become prohibitively expensive. Here we aim to reduce the costs of Transformers by searching for a more efficient variant.

**ArXiv:** [2109.08668](https://arxiv.org/abs/2109.08668),[Local link](Papers/2109.08668v2_primer_llm_architecture_search.pdf)

#### 23. Rare Gems: Finding Lottery Tickets at Initialization

*Kartik Sreenivasan, Jy-yong Sohn, Liu Yang, Matthew Grindenwalt, Alliot Nagle, Hongyi Wang, Eric Xing, Kangwook Lee, Dimitris Papailiopoulos*

**Summary:** This paper introduces GEM-MINER, a method that finds lottery tickets at initialization which can be trained to beat current baselines in accuracy and is up to 19 times faster than Iterative Magnitude Pruning (IMP).

**ArXiv:** [2202.12002](https://arxiv.org/abs/2202.12002),[Local link](Papers/2202.12002v2_Finding Lottery Tickets at Initialization.pdf)

#### 24. Optimal Clipping and Magnitude-aware Differentiation for Improved Quantization-aware Training

*Charbel Sakr, Steve Dai, Rangharajan Venkatesan, Brian Zimmer, William J. Dally, Brucek Khailany*

**Summary:** This paper proposes Optimally Clipped Tensors And Vectors (OCTA V), a recursive algorithm to determine MSE-optimal clipping scalars for reduced quantization noise in quantization-aware training (QAT). Additionally, it reveals limitations in common gradient estimation techniques and proposes magnitude-aware differentiation as a remedy.

**ArXiv:** [2206.06501](https://arxiv.org/abs/2206.06501),[Local link](Papers/2206.06501v1_optimal_clipping.pdf)

#### 25. LLM.int8() : 8-bit Matrix Multiplication for Transformers at Scale

*Tim Dettmers, Mike Lewis, Younes Belkada, Luke Zettlemoyer*

**Summary:** Large language models have been widely adopted but require significant GPU memory for inference. We develop a procedure for Int8 matrix multiplication for feed-forward and attention projection layers in transformers, which cut the memory needed for inference by half while retaining full precision performance.

**ArXiv:** [2208.07339](https://arxiv.org/abs/2208.07339),[Local link](Papers/2208.07339.pdf)

#### 26. GPTQ: A CURATE POST-TRAINING QUANTIZATION FOR GENERATIVE PRE-TRAINED TRANSFORMERS

*Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh*

**Summary:** Generative Pre-trained Transformer models, known as GPT or OPT, set themselves apart through breakthrough performance across complex language modelling tasks, but also by their extremely high computational and storage costs. Specifically, due to their massive size, even inference for large, highly-accurate GPT models may require multiple performant GPUs, which limits the usability of such models. While there is emerging work on relieving this pressure via model compression, the applicability and performance of existing compression techniques is limited by the scale and complexity of GPT models.

**ArXiv:** [2210.17323](https://arxiv.org/abs/2210.17323),[Local link](Papers/2210.17323.pdf)

#### 27. GPTQ: A CURATE POST-TRAINING QUANTIZATION FOR GENERATIVE PRE-TRAINED TRANSFORMERS

*Elias Frantar, Saleh Ashkboos, Torsten Hoefler, Dan Alistarh*

**Summary:** GPTQ, a new one-shot weight quantization method based on approximate second-order information, can accurately and efficiently compress GPT models, allowing their execution inside a single GPU for generative inference.

**ArXiv:** [2210.17323](https://arxiv.org/abs/2210.17323),[Local link](Papers/2210.17323v2_GPTQ.pdf)

#### 28. Symbolic Discovery of Optimization Algorithms

*Xiangning Chen, Chen Liang, Da Huang, Esteban Real, Kaiyuan Wang, Yao Liu, Hieu Pham, Xuanyi Dong, Thang Luong, Cho-Jui Hsieh, Yifeng Lu, Quoc V. Le*

**Summary:** We present a method to formulate algorithm discovery as program search, and apply it to discover optimization algorithms for deep neural network training.

**ArXiv:** [2302.06675](https://arxiv.org/abs/2302.06675),[Local link](Papers/2302.06675v4_Lion_Opitmizer.pdf)

#### 29. Multiplication-Free Transformer Training via Piecewise Affine Operations

*Atli Kosson, Martin Jaggi*

**Summary:** We show that transformers can be trained with a cheap piecewise affine approximation of floating point numbers, and without changes to the training hyperparameters.

**ArXiv:** [2305.17190](https://arxiv.org/abs/2305.17190),[Local link](Papers/2305.17190v2_mulfree.pdf)

#### 30. A Activation-aware Weight Quantization for LLM Compression and Acceleration

*Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Xingyu Dang, Chuang Gan, Song Han*

**Summary:** This paper proposes Activation-aware Weight Quantization (AWQ), a hardware-friendly approach for Large Language Model low-bit weight-only quantization, which can greatly reduce quantization error by protecting only 1% of salient weights.

**ArXiv:** [2306.00978](https://arxiv.org/abs/2306.00978),[Local link](Papers/2306.00978.pdf)

#### 31. Patch n’ Pack: NaViT, a Vision Transformer for any Aspect Ratio and Resolution

*Mostafa Dehghani, Basil Mustafa, Josip Djolonga, Jonathan Heek, Matthias Minderer, Mathilde Caron, Andreas Steiner, Joan Puigcerver, Robert Geirhos, Ibrahim Alabdulmohsin Avital, Oliver Piotr Padlewski, Alexey Gritsenko, Mario Lučić, Neil Houlsby*

**Summary:** The ubiquitous and demonstrably suboptimal choice of resizing images to a fixed resolution before processing them with computer vision models has not yet been successfully challenged. However, models such as the Vision Transformer (ViT) offer flexible sequence-based modeling, and hence varying input sequence lengths. We take advantage of this with NaViT(Native Resolution ViT) which uses sequence packing during training to process inputs of arbitrary resolutions and aspect ratios.

**ArXiv:** [2307.06304](https://arxiv.org/abs/2307.06304),[Local link](Papers/2307.06304.pdf)

#### 32. Explaining grokking through circuit efficiency

*Vikrant Varma, Rohin Shah, Zachary Kenton, János Kramár, Ramana Kumar*

**Summary:** One of the most surprising puzzles in neural network generalisation is grokking: a network with perfect training accuracy but poor generalisation will, upon further training, transition to perfect generalisation.

**ArXiv:** [2309.02390](https://arxiv.org/abs/2309.02390),[Local link](Papers/2309.02390v1_grokking.pdf)

#### 33. BitNet: Scaling 1-bit Transformers for Large Language Models

*Hongyu Wang, Shuming Ma, Li Dong, Shaohan Huang, Huaijie Wang, Lingxiao Ma, Fan Yang, Ruiping Wang, Yi Wu, Furu Wei*

**Summary:** The increasing size of large language models has posed challenges for deployment and raised concerns about environmental impact due to high energy consumption. In this work, we introduce BitNet, a scalable and stable 1-bit Transformer architecture designed for large language models.

**ArXiv:** [2310.11453](https://arxiv.org/abs/2310.11453),[Local link](Papers/2310.11453_bitnet.pdf)

#### 34. FP6-LLM: Efficiently Serving Large Language Models Through FP6-Centric Algorithm-System Co-Design

*Haojun Xia, Zhen Zheng, Xiaoxia Wu, Shiyang Chen, Zhewei Yao, Stephen Youn, Arash Bakhtiari, Michael Wyatt, Donglin Zhuang, Zhongzhu Zhou, Olatunji Ruwase, Yuxiong He, Shuaiwen Leon Song*

**Summary:** Six-bit quantization (FP6) can effectively reduce the size of large language models (LLMs) and preserve the model quality consistently across varied applications. However, existing systems do not provide Tensor Core support for FP6 quantization and struggle to achieve practical performance improvements during LLM inference.

**ArXiv:** [2401.14112](https://arxiv.org/abs/2401.14112),[Local link](Papers/2401.14112v2_fp6_llm.pdf)

#### 35. No Free Prune: Information-Theoretic Barriers to Pruning at Initialization

*Tanishq Kumar, Kevin Luo, Mark Sellke*

**Summary:** We put forward a theoretical explanation for the failure of pruning neural networks at initialization based on the model's effective parameter count, peff, given by the sum of the number of non-zero weights in the final network and the mutual information between the sparsity mask and the data.

**ArXiv:** [2402.01089](https://arxiv.org/abs/2402.01089),[Local link](Papers/2402.01089v1_No Free Prune.pdf)

#### 36. MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases

*Zechun Liu, Changsheng Zhao, Forrest Iandola, Chen Lai, Yuandong Tian, Igor Fedorov, Yunyang Xiong, Ernie Chang, Yangyang Shi, Raghuraman Krishnamoorthi, Liangzhen Lai, Vikas Chandra*

**Summary:** This paper addresses the growing need for efficient large language models (LLMs) on mobile devices, driven by increasing cloud costs and latency concerns. We focus on designing top-quality LLMs with fewer than a billion parameters, a practical choice for mobile deployment.

**ArXiv:** [2307.16152](https://arxiv.org/abs/2307.16152),[Local link](Papers/2402.14905v2_mobilellm.pdf)

#### 37. The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits

*Shuming Ma, Hongyu Wang, Lingxiao Ma, Lei Wang, Wenhui Wang, Shaohan Huang, Li Dong, Ruiping Wang, Jilong Xue, Furu Wei*

**Summary:** Recent research, such as BitNet [ WMD+23], is paving the way for a new era of 1- bit Large Language Models (LLMs). In this work, we introduce a 1-bit LLM variant, namely BitNet b1.58 , in which every single parameter (or weight) of the LLM is ternary {-1, 0, 1}. It matches the full-precision (i.e., FP16 or BF16) Transformer LLM with the same model size and training tokens in terms of both perplexity and end-task performance, while being significantly more cost-effective in terms of latency, memory, throughput, and energy consumption.

**ArXiv:** [2402.17764](https://arxiv.org/abs/2402.17764),[Local link](Papers/2402.17764_1_58bit.pdf)

#### 38. The Unreasonable Ineffectiveness of the Deeper Layers

*Andrey Gromov, Kushal Tirumala, Hassan Shapourian, Paolo Glorioso, Daniel A. Roberts*

**Summary:** We empirically study a simple layer-pruning strategy for popular families of open-weight pretrained LLMs, finding minimal degradation of performance on different question-answering benchmarks until after a large fraction (up to half) of the layers are removed.

**ArXiv:** [2403.17887](https://arxiv.org/abs/2403.17887),[Local link](Papers/2403.17887v1_layer_pruning.pdf)

#### 39. Mixture-of-Depths: Dynamically allocating compute in transformer-based language models

*David Raposo, Sam Ritter, Blake Richards, Timothy Lillicrap, Peter Conway Humphreys, Adam Santoro*

**Summary:** In this work, the authors demonstrate that transformers can learn to dynamically allocate FLOPs (or compute) to specific positions in a sequence, optimizing the allocation along the sequence for different layers across the model depth.

**ArXiv:** [2404.02258](https://arxiv.org/abs/2404.02258),[Local link](Papers/2404.02258.pdf)

#### 40. Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction

*Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, Liwei Wang*

**Summary:** We present Visual AutoRegressive modeling (VAR), a new generation paradigm that redefines the autoregressive learning on images as coarse-to-fine "next-scale prediction" or "next-resolution prediction", diverging from the standard raster-scan "next-token prediction".

**ArXiv:** [2404.02905](https://arxiv.org/abs/2404.02905),[Local link](Papers/2404.02905.pdf)

#### 41. Talaria: Interactively Optimizing Machine Learning Models for Efficient Inference

*Fred Hohman, Chaoqun Wang, Jinmook Lee, Jochen Görtler, Dominik Moritz, Jeffrey P. Bigham, Zhile Ren, Cecile Foret, Qi Shan, Xiaoyi Zhang*

**Summary:** On-device machine learning (ML) moves computation from the cloud to personal devices, protecting user privacy and enabling in-telligent user experiences. However, fitting models on devices with limited resources can be challenging.

**ArXiv:** [2404.03085](https://arxiv.org/abs/2404.03085),[Local link](Papers/2404.03085v1_talaria.pdf)

#### 42. Training LLMs over Neurally Compressed Text

*Brian Lester, Jaehoon Lee, Alex Alemi, Jeffrey Pennington, Adam Roberts, Jascha Sohl-Dickstein, Noah Constant*

**Summary:** In this paper, we explore the idea of training large language models (LLMs) over highly compressed text using Equal-Info Windows compression technique.

**ArXiv:** [2404.03626](https://arxiv.org/abs/2404.03626),[Local link](Papers/2404.03626.pdf)

#### 43. Physics of Language Models: Part 3.3 Knowledge Capacity Scaling Laws

*Zeyuan Allen-Zhu, Yuanzhi Li*

**Summary:** Scaling laws describe the relationship between the size of language models and their capabilities. Unlike prior studies that evaluate a model's capability via loss or benchmarks, we estimate the number of knowledge bits a model stores.

**ArXiv:** [2404.05405](https://arxiv.org/abs/2404.05405),[Local link](Papers/2404.05405v1_llm_scaling_meta.pdf)

#### 44. Physics of Language Models: Part 3.3 Knowledge Capacity Scaling Laws

*Zeyuan Allen-Zhu, Yuanzhi Li*

**Summary:** Scaling laws describe the relationship between the size of language models and their capabilities. Unlike prior studies that evaluate a model's capability via loss or benchmarks, we estimate the number of knowledge bits a model stores.

**ArXiv:** [2404.05405](https://arxiv.org/abs/2404.05405),[Local link](Papers/2404.05405v1_physics_of_llm.pdf)

#### 45. Grokked Transformers are Implicit Reasoners: A Mechanistic Journey to the Edge of Generalization

*Boshi Wang, Xiang Yue, Yu Su, Huan Sun*

**Summary:** We study whether transformers can learn to implicitly reason over parametric knowledge, a skill that even the most capable language models struggle with. Focusing on two representative reasoning types, composition and comparison, we consistently find that transformers canlearn implicit reasoning, but only through grokking , i.e., extended training far beyond overfitting.

**ArXiv:** [2405.15071](https://arxiv.org/abs/2405.15071),[Local link](Papers/2405.15071v1_grokking.pdf)

#### 46. Scalable MatMul-free Language Modeling

*Rui-Jie Zhu, Yu Zhang, Ethan Sifferman, Tyler Sheaves, Yiqiao Wang, Dustin Richmond, Peng Zhou*

**Summary:** This work demonstrates that matrix multiplication (MatMul) operations can be completely eliminated from large language models (LLMs) while maintaining strong performance at billion-parameter scales.

**ArXiv:** [2406.02528](https://arxiv.org/abs/2406.02528),[Local link](Papers/2406.02528v1_matmul_free.pdf)

#### 47. Q-Sparse: All Large Language Models can be Fully Sparsely-Activated

*Hongyu Wang, Shuming Ma, Ruiping Wang, Furu Wei*

**Summary:** We introduce, Q-Sparse, a simple yet effective approach to training sparsely-activated large language models (LLMs). Q-Sparse enables full sparsity of activations in LLMs which can bring significant efficiency gains in inference.

**ArXiv:** [2407.10969](https://arxiv.org/abs/2407.10969),[Local link](Papers/2407.10969v1_Q-sparse.pdf)

#### 48. Least Squares Quantization in PCM

*S. P. Lloyd*

**Summary:** Necessary conditions are found that the quanta and associated quantization intervals of an optimum finite quantization scheme must satisfy, with optimization criterion being that the average quantization noise power is a minimum.

**ArXiv:** N/A,[Local link](Papers/lloyd1982_least_squares_quant.pdf)

#### 49. ShiftAddNet: A Hardware-Inspired Deep Network

*Haoran You, Xiaohan Chen, Yongan Zhang, Chaojian Li, Sicheng Li, Zihao Liu, Zhangyang Wang, Yingyan Lin*

**Summary:** This paper introduces ShiftAddNet, a deep neural network inspired by energy-efficient hardware implementation that uses only bit-shift and additive weight layers to reduce resource costs for deployment on edge devices.

**ArXiv:** [1905.13298](https://arxiv.org/abs/1905.13298),[Local link](Papers/NeurIPS-2020-shiftaddnet-a-hardware-inspired-deep-network-Paper.pdf)

#### 50. Optimal Brain Damage

*Yann Le Cun, John S. Denker, Sara A. Solla*

**Summary:** We have used information-theoretic ideas to derive a class of practical and nearly optimal schemes for adapting the size of a neural network.

**ArXiv:** N/A,[Local link](Papers/NIPS-1989-optimal-brain-damage-Paper.pdf)

#### 51. OCP Micro scaling Format s (MX) Specification

*Bita Darvish Rouhani, Nitin Garegrat, Tom Savell , Ankit More, Kyung -Nam Han, Ritchie Zhao, Mathew Hall, Jasmine Klar, Eric Chung, Yuan Yu, Microsoft, Michael Schulte, Ralph Wittig, AMD, Ian Bratt, Nigel Stephens, Jelena Milanovic, John Brothers, Arm, Pradeep Dubey, Marius Cornea, Alexander Heinecke, Andres Rodriguez, Martin Langhammer, Intel, Summer Deng, Maxim Naumov, Meta, Paulius Micikevicius, Michael Siu, NVIDI A, Colin Verrilli, Qualcomm*

**Summary:** The Open Compute Project (OCP) Micro scaling Formats (MX) Specification provides definitions and specifications for various formats that can be used in microscaling operations.

**ArXiv:** N/A,[Local link](Papers/OCP_Microscaling Formats (MX) v1.0 Spec_Final.pdf)

