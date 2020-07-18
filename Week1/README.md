# What are Channels and Kernels (according to EVA)?
* Channels are the homogeneous collection of features. Feature is the unique property of the channel.
* Kernels are the feature extractors which operate on top the image to extract the channel. In DNN kernels are represented by randomly initialized 3x3 matirces.
* Example: In an Opera playing music with multiple instruments, each instrument represents a particular sound (Channel). A composer (Kernel) can  separate the different notes from the symphony and each unique note from the instrument (channel) is the feature of that instrument.
![](Symphony.PNG)

# Why should we (nearly) always use 3x3 kernels?
* Convolutions using other kernels both lower order and high order are possible. (2x2, 4x4, 5x5 etc). But using 3x3 gives a unique advantage to the DNN.
* Convolutions using 1x1 kernel isn't useful in detecting edges and gradients (step 1 in DNN).
* Although 2x2 Kernels could be used for convolutions. It lacks a central axis (Symmetry) provided by the 3x3. The Central axis accentuates the gradient amplitude of the edges, hence giving better results. 
* Now, any Kernel with higher odd order could provide this benefit, but its performance is attenuated by the fact that more computations are required to be done during each stage of convolution.
  * 3x3 - 9 variables initialized
  * 5x5 - 25 variables initialized
  * 7x7 - 49 variables initialized
* Though using higher order kernels can reduce the number of layers. The exponential increase in the shear number of computations required at each layers outweighs the benefits.

 # How many times to we need to perform 3x3 convolutions operations to reach close to 1x1 from 199x199 (type each layer output like 199x199 > 197x197...)

* 99 convolutions are to be performed.
```
Input	KernelUsed Output No.ofConvolutions
199x199	3x3	197x197	1
197x197	3x3	195x195	2
195x195	3x3	193x193	3
193x193	3x3	191x191	4
191x191	3x3	189x189	5
189x189	3x3	187x187	6
187x187	3x3	185x185	7
185x185	3x3	183x183	8
183x183	3x3	181x181	9
181x181	3x3	179x179	10
179x179	3x3	177x177	11
177x177	3x3	175x175	12
175x175	3x3	173x173	13
173x173	3x3	171x171	14
171x171	3x3	169x169	15
169x169	3x3	167x167	16
167x167	3x3	165x165	17
165x165	3x3	163x163	18
163x163	3x3	161x161	19
161x161	3x3	159x159	20
159x159	3x3	157x157	21
157x157	3x3	155x155	22
155x155	3x3	153x153	23
153x153	3x3	151x151	24
151x151	3x3	149x149	25
149x149	3x3	147x147	26
147x147	3x3	145x145	27
145x145	3x3	143x143	28
143x143	3x3	141x141	29
141x141	3x3	139x139	30
139x139	3x3	137x137	31
137x137	3x3	135x135	32
135x135	3x3	133x133	33
133x133	3x3	131x131	34
131x131	3x3	129x129	35
129x129	3x3	127x127	36
127x127	3x3	125x125	37
125x125	3x3	123x123	38
123x123	3x3	121x121	39
121x121	3x3	119x119	40
119x119	3x3	117x117	41
117x117	3x3	115x115	42
115x115	3x3	113x113	43
113x113	3x3	111x111	44
111x111	3x3	109x109	45
109x109	3x3	107x107	46
107x107	3x3	105x105	47
105x105	3x3	103x103	48
103x103	3x3	101x101	49
101x101	3x3	99x99	50
99x99	3x3	97x97	51
97x97	3x3	95x95	52
95x95	3x3	93x93	53
93x93	3x3	91x91	54
91x91	3x3	89x89	55
89x89	3x3	87x87	56
87x87	3x3	85x85	57
85x85	3x3	83x83	58
83x83	3x3	81x81	59
81x81	3x3	79x79	60
79x79	3x3	77x77	61
77x77	3x3	75x75	62
75x75	3x3	73x73	63
73x73	3x3	71x71	64
71x71	3x3	69x69	65
69x69	3x3	67x67	66
67x67	3x3	65x65	67
65x65	3x3	63x63	68
63x63	3x3	61x61	69
61x61	3x3	59x59	70
59x59	3x3	57x57	71
57x57	3x3	55x55	72
55x55	3x3	53x53	73
53x53	3x3	51x51	74
51x51	3x3	49x49	75
49x49	3x3	47x47	76
47x47	3x3	45x45	77
45x45	3x3	43x43	78
43x43	3x3	41x41	79
41x41	3x3	39x39	80
39x39	3x3	37x37	81
37x37	3x3	35x35	82
35x35	3x3	33x33	83
33x33	3x3	31x31	84
31x31	3x3	29x29	85
29x29	3x3	27x27	86
27x27	3x3	25x25	87
25x25	3x3	23x23	88
23x23	3x3	21x21	89
21x21	3x3	19x19	90
19x19	3x3	17x17	91
17x17	3x3	15x15	92
15x15	3x3	13x13	93
13x13	3x3	11x11	94
11x11	3x3	9x9	95
9x9	3x3	7x7	96
7x7	3x3	5x5	97
5x5	3x3	3x3	98
3x3	3x3	1x1	99
```
# How are kernels initialized?
* Kernels in a neural network are randomly initialized between 0 and 1.
```
import numpy as np
kernel=np.random.rand(3,3)
print(kernel)

Output:
[[0.7591608, 0.77821914 0.08635577]
[0.21025328, 0.19547953 0.3046247]
[0.723433, 0.32978227 0.31589996]]
```
# What happens during the training of a DNN?
* When the input images are passed through the DNN, kernels are randomly initialized and the network will try to extract edges and gradients, then textures and patterns, and build parts of object and then finally objects. The model is punished or rewarded based on the output and the predicted classes using Back Propagation.
![](BasicDNN.PNG)
