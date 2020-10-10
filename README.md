# Model_Quantization
The basic methods for model quantization


***What is quantization?***

It typically involves:


    1. A transfer function to go from high to lower precision (e.g. affine)
    2. A conversion/legalization(转换) process from program p to p'
    3. A calibration algorithm (校准) to compute data (e.g. scaling factors) required by the new program p', and potentially fine-tuning of the original program's parameters as they will be used in p'


***Why is it hard?***

    Reason one: (硬件要求, 加速, 高效)
    Hardware programmability and preferences: let it be due to the hardware's design (e.g. TPU) or for efficiency reasons (e.g. CPU/GPU), the set of operations available to expres the original program is restricted. 

    Reason two: (需要更多的数据和分析)
    The program may not be enough: depending on the procedure used, there may be extra data/metadata needed to complete the transformation (e.g. execution statistics, or even labeled data for finetuning).

    Reason three: (模型可解释性差)
    ML program interpretability: not being able to interpret an ML program, and thus the effects of modifying it, hinders the transformations that can be safely applied without catastrophic consequences.

***What is the SOTA?***

    1. Affine representation:

    It linearly maps values expressed as a transormation involving as a scale and an offset(or zero point) on 1 or more dimensions (or even slices) of a Tensor. Typical hyperparameters are the number of bits used to represent the values and whether they are symmetrically or asymmetrically distributed. 

    We need to know min and max value to know how to quantize. 

    It is tricky to clip these extreme values outside min-max since it's hard to evalue the importance of those values.

    DSP - block floating-point - the exponents shared across all values of a tensor

    2. Conversion and calibration

    Currently, calibration can be seen as either a process that happens either during or post training the ML program, and it's tightly coupled with the program's conversion/ legalization. 

    2.1 During training 

    Attempts to emulate the quantized program's inference, given a quantization function and an assumed form in which it will be legalized. Output is a float program with extra logging ops to transform into p'.
    Goal: train as closely what you'll execute to allow parameters to adjust
    Caveats: hard to author programs, unstable convergence, hard to reuse/retrain program (fine-tuned to a target).

    2.2 After training

    Skips fine-tuning the program's parameters, and instead just legalizes it to p'(given a quantization function and assumed target support). Depending on the algorithm, p' may require extra datas such as scaling factors for which it is necessary to run inferences and collect statistics.
    Goal: enable simple quantization of any existing program, with no required knowledge and little or no data.
    Caveats: in some cases accuracy is lower, particularly for <8bit


Quantization specification

- float32 --> int8

- scale factor:\
    scale = (max - min)/ 2^{bits}

- zero-point:\
 An integer value that corresponds to floating point zero

- Quantized_value = float_value/scale + zero_point
- float_value = (quantized_value - zero_point)*scale

Symmetric:
- Per-axis symmetric weights represented by int8 two's complement values in the range [-127, 127]
  
- Per-layer asymmetric activations represented by int8 two's complement values in the range [-128, 127]
  
- Zero-point in range [-128, 127]

- Dequantize: real_value = (int8_value - zero_point)*scale

Weight is better to be quantized symmetric/ Activation is better to be quantized asymmetric



Per layer / Per axis

For each Conv/Depthwise channel, weights have different distributions. 

- Each (weight) tensor axis is quantized separately 
- Gets most bang for the buck for CNN accuracy








