## Blog Post Title From First Header

Autoencoders (AEs) are a family of generative models which make use of the encoder-decoder architecture to model data and reconstruct varied forms or representations of the input distribution. This aspect propels the autoencoder family into the generative AI sphere.

For those curious to learn more about what generative AI is, I penned down a gripping introductory blog which can be found here🤓

The encoding mechanism enables an autoencoder to compress the original data and its dimensions to a latent space. The lower dimensional space acts as a bottleneck to later aid the decoder in reconstructing the data.

💡It is interesting to note that autoencoders employ unsupervised learning and search for structure in the data via compressed representation to make sense of their inputs.
---

### Some Math

Let’s look at the mathematical notations involved:

x is the input image of number 4 (from the MNIST dataset) represented as a matrix of pixel values.
The encoder function gφ() compresses the input x (reduces its dimensionality) to form the compact bottleneck z.
The bottleneck z = gφ(x) helps the network learn the input structure instead of overfitting by merely memorizing the input data for reconstruction. Nonetheless, autoencoders possess this drawback.
The decoder function fθ() reconstructs image data (or the number 4 in our case) resembling the input, from the latent representation. It outputs x’ = fθ(z).

💡 The low dimensional representation in the bottleneck is also referred to as latent code.

The loss function here is simply the measurement of error (or difference) between the original input image x and the reconstructed output image x’ represented by fθ(gφ(x)).

Loss(θ, φ) = 1/n . Σ [x_i — fθ(gφ(x_i))]²

The parameters φ and θ are learnable through backpropagation during training. They help our network produce outputs similar to the input. But beware, as already mentioned above in point 3, sometimes the network may end up memorizing the input data points due to overfitting towards the identity function. Thus it will get stuck reproducing images not very unlike the input training data sample.

### Denoising Autoencoders (DAEs)🔇
Since overfitting to the identity function is a fairly well-known drawback of autoencoders, denoising autoencoders (DAEs) were introduced to rectify this issue. How is a DAE different? It corrupts the input data (noises up the network) by canceling out certain input values sporadically.

So, we simply set the values of a few input dimensions to 0. This forces the network to “denoise” the corrupted inputs. Hence the reconstructed data is similar to the original inputs but the model does not overfit.

This is because DAEs learn from a corrupted (partially visible input) distribution as compared to merely memorizing the input down to a tee. This indicates successful data retrieval and reconstruction from the latent distribution.

💡The loss function must compare the network’s outputs only with the original uncorrupted inputs.

### Add a V to the AE — Variational Autoencoders (VAEs)
Autoencoders use fixed vectors to map attributes from the input image to the latent space. This cripples the network’s ability to better learn, customize, or tweak certain input attributes.

For example, in our input image say we have a woman with blue hair highlights. A simple autoencoder maps a fixed vector (a single value) to the “blue hair” attribute. It can probably only turn the blue hair on or off. But this is not the best way to quantify a transient attribute such as hair color.

On the other hand, variational autoencoders map a probability distribution (a range of values) for better control over the network’s latent representations as compared to one rigid latent vector in autoencoders for representing complex attributes.

Imagine we are comparing an on/off switch with a rotating dial that you can twist for better control. VAEs resemble the more controllable and flexible dial. This also helps take into consideration the nuances in our data.

The VAE consists of generative and inference models for the tasks of compressing and reconstructing data. Let x be our observed variable (or input) and z be our latent variable. Our goal is to infer the latent variable from our input. For that we need to determine the conditional probability distribution of the latent variable given the input i.e. we need to find pθ(z|x).

#### Why variational inference?🤔
Here, while computing the probability distribution pθ(z|x) for the input, we use the Bayes rule to expand:

Equation A → pθ(z|x) = {pθ(x|z) . pθ(z)} ÷ {pθ(x)}

Here,

pθ(z|x) is the posterior
pθ(x|z) is the likelihood
pθ(z) is the prior and
pθ(x) is the marginal
We need to compute the marginal pθ(x) but this is arduous since it integrates over all the values of z:

pθ(x) = ∫ pθ(x | z) . pθ(z) dz

This computation is intractable (no efficient solution exists) in higher dimensional spaces as z being a latent space can have numerous values and dimensions. It requires exponential time for solving which is not plausible. Thus we resort to variational inference.

#### Variational inference😃
By making use of another known and tractable yet simple Gaussian distribution qφ(z|x), we approximate pθ(x|z). So we need to ensure that pθ(z|x) is modeled from qφ(z|x) by converting this into an optimization problem.

We can achieve this with the Kullback-Leibler (KL) divergence measure. KL divergence tells us how different the two probability distributions are. Let’s take a look at the equation for KL divergence for a discrete random variable and two distributions say A and B.

KL (A||B) = Σ A(xi) . log(A(xi) ÷ B(xi))

💡KL(A||B) ≠ KL(B||A) — the forward and reverse KL divergence measures are not equal and it is asymmetric in nature.

The KL divergence measure is non-negative and if the 2 distributions are mostly equal throughout, it amounts to zero. So for our 2 distributions p and q, we get:

KL(qφ(z|x)||pθ(z|x)) = Σ qφ(z|x) . log (qφ(z|x) ÷ pθ(z|x))

### Loss function derivation🪄
In terms of the expectation and after some logarithmic simplification, the RHS becomes:

Equation B → KL = E_z [ log (qφ(z|x)) — log( Pθ(z|x))]

In equation B,

1. KL = KL( qφ(z|x) || pθ(z|x) )

2. E_z signifies the expectation over z wherein z is being sampled from qφ(z|x) and can also be written as → z ~ qφ(z|x).

Substituting equation A in equation B, we get:

→ KL = E_z [ log (qφ(z|x)) — log( {pθ(x|z) . pθ(z)} ÷ {pθ(x)} ) ]

→ KL = E_z [ log (qφ(z|x)) — log(pθ(x|z)) — log(pθ(z)) + log(pθ(x)) ]

Although we still don’t know pθ(x), note that the expectation term only computes over z and doesn’t involve x. So we can move the term log(pθ(x)) to the LHS instead of retaining it in the RHS.

→ KL — log(pθ(x)) = E_z [ log (qφ(z|x)) — log(pθ(x|z)) — log (pθ(z)) ]

Taking a minus sign on both sides and rearranging, we get,

→ log(pθ(x)) — KL = E_z[ log(pθ(x|z)) ] — E_z [ log (qφ(z|x)) — log (pθ(z)) ]

→ log(pθ(x)) — KL = E_z[ log(pθ(x|z)) ] — E_z [ log ( qφ(z|x) ÷ pθ(z) ) ]

Note that the second term (bolded) on the RHS is another KL divergence measure. Hence, we get:

Equation C → log(pθ(x)) — KL = E_z[ log(pθ(x|z)) ] — KL ( qφ(z|x) || (pθ(z) )

To arrive at the final loss function of a VAE, we attach a negative sign to the RHS of equation C:

L(θ, φ) = -E_z [log (pθ(x|z))]+ KL[ qφ(z|x) || pθ(z) ]

Also, the LHS of equation C can be represented as:

→ log(pθ(x)) — KL [ qφ(z|x) || pθ(z|x) ] = — L(θ, φ)

In the final loss function, the roles of both terms are as follows:

The first term is the reconstruction loss.
The second term is considered a regularizer term.


#### The Reparameterization trick😎
As we know, the expectation term E_z represents z being sampled from qφ(z|x) and is written as z ~ qφ(z|x). Unfortunately, the sampling of z is a stochastic process, so we are dealing with a random variable. Random variables don’t allow us to calculate their derivatives and hence aren’t backpropagation-friendly.

This is a problem because we can’t train our network with random variables that can’t be backpropagated. Hence we use the reparameterization trick: represent the random variable as a deterministic variable through a transformation function gφ().

z = gφ (ε, x) and z = µ + σ.ε

Here, ε is an extra stochastic random variable that allows z to become learnable due to the presence of mean µ and variance σ parameters, hence making it training-friendly🤩

With this, we can better comprehend what goes on behind the scenes in a VAE. What’s left is to build this model using a deep learning framework, train it, and later use it for inference purposes. For this purpose, refer to my GitHub code repository below.

You can find my PyTorch-based VAE implementation here:

### Vector Quantized VAEs (VQ-VAEs)
A VQ-VAE employs vector quantized representations of latent spaces and its encoder outputs “discrete” codes to model and then the input data is compressed. Further, the prior distribution is learned and not fixed.

The authors of the VQ-VAE paper decided to incorporate discrete representations as they more accurately resemble naturally available data such as language which is a sequence of characters and images that can be described by words, etc. Through this method, drawbacks such as posterior collapse and exploding variance are alleviated.

The image above depicts how data flows through the entire network(both forward and backward). We have a convolutional neural network-based (CNN) encoder and decoder. Additionally, we have the vector quantization block between the encoder and decoder which uses an embedding space with embedding vectors e_i ∈ ℝ where i ∈ 1,2,...K.

#### Forward propagation➡️
Encoder: It uses a CNN to encode the input data (x) and gives out: z_e(x) of size D which is the embedding size.

We then calculate the one-hot posterior q(z|x) distribution. Discrete latent variables z are found using nearest neighbor lookup to match with one of k embedding vectors from the embedding space.

💡k is assigned according to the L2 norm calculated between the encoder’s output and embedding vector. This helps ascertain the embedding element nearest to z_e(x) i.e. the least distance between them.

- The term q(z|x) is indexed using the codebook to extract embedding vectors where integer k is the common link connecting the z_e(x) to z_q(x).

- We then find the embedding vector (represented as e_k) nearest to z_e(x) for ensuring dimensional uniformity and assign it to z_q(x).

- Now, z_q(x) is passed on to the decoder network for image reconstruction.

### Backpropagation: problems and solution⬅️

The codebook vector is not differentiable — gradients can’t be propagated back through argmin as it maps k to discrete integers (indices of the closest vectors). Thus we use a straight-through estimation mechanism that copies the gradients of z_q(x) into z_e(x) since both vectors have the same dimensions. This is represented as a red line in the VQ-VAE: Working Mechanism image above to indicate straight-through gradient estimation. So now, we can go back😉

Loss function
Here we have the final training loss or objective:


We have 3 terms, all separated by plus (+) signs in the equation above. Here, sg is the stop gradient constraint or operator which signifies the absence of any gradients for the term it encompasses and uses a non-upgradable constant. So let’s break the equation down and tackle each term.

- First term log p(x|z_q(x)) → The reconstruction loss optimizes both the encoder and decoder networks.
- Second term ||sg[z_e(x)] — e||2 2 → The Vector Quantization (VQ) loss is also called the codebook objective. The L2 error is used to move the embedding vector e_i closer to the encoded outputs z_e(x). This term helps learn and update the embedding space.
- Third term β||z_e(x)−sg[e]||2 2 → The commitment loss function ensures that the encoder’s outputs don’t vacillate or grow too much and stay close to the embedding space.

💡VQ-VAEs achieve commendable likelihood for generating 128x128 color images, speech, and video action sequences too!

You can find an example of a VQ-VAE’s performance with 32x32x1 latent space and k=512 attached as an image below. This has been obtained from the original research paper by Oord et. al. and can be found here. The left half consists of original images from ImageNet and the right half shows the model’s reconstructed versions of the same.

In this blog, we have explored, analyzed, and understood how autoencoders have been evolving with new additions and conceptual improvements that enhance their generative capabilities.

### Conclusion😄
OK, you have successfully gotten through this detailed and hopefully informative blog! What’s next? I will be releasing the third blog in this series soon. It will be all about Generative Adversarial Networks (GANs) — the concepts used to build them, some useful math to understand their working, etc. Stay tuned!🫡

Until then, if you have any doubts/suggestions, or would simply like to chat, feel free to reach out to me via LinkedIn.

