Autoencoders (AEs) are a family of generative models which make use of the encoder-decoder architecture to model data and reconstruct varied forms or representations of the input distribution. This aspect propels the autoencoder family into the generative AI sphere.

For those curious to learn more about what generative AI is, I penned down an introductory blog which can be found [here](post.html?post=genai-intro).

The encoding mechanism enables an autoencoder to compress the original data and its dimensions to a latent space. The lower dimensional space acts as a bottleneck to later aid the decoder in reconstructing the data.

> It is interesting to note that autoencoders employ unsupervised learning and search for structure in the data via compressed representation to make sense of their inputs.

Let's look at the mathematical notations involved:

1. *x* is the input image of number 4 (from the MNIST dataset) represented as a matrix of pixel values.
2. The encoder function *g_phi()* compresses the input *x* (reduces its dimensionality) to form the compact bottleneck *z*.
3. The bottleneck *z = g_phi(x)* helps the network learn the input structure instead of overfitting by merely memorizing the input data for reconstruction. Nonetheless, autoencoders possess this drawback.
4. The decoder function *f_theta()* reconstructs image data resembling the input, from the latent representation. It outputs *x' = f_theta(z)*.

> The low dimensional representation in the bottleneck is also referred to as latent code.

The loss function here is simply the measurement of error (or difference) between the original input image *x* and the reconstructed output image *x'* represented by **f_theta(g_phi(x))**.

> **Loss(theta, phi) = 1/n . Sigma [x_i - f_theta(g_phi(x_i))]^2**

The parameters *phi* and *theta* are learnable through backpropagation during training. They help our network produce outputs similar to the input. But beware, sometimes the network may end up memorizing the input data points due to overfitting towards the identity function. Thus it will get stuck reproducing images not very unlike the input training data sample.

---

## Denoising Autoencoders (DAEs)

Since overfitting to the identity function is a fairly well-known drawback of autoencoders, denoising autoencoders (DAEs) were introduced to rectify this issue. How is a DAE different? It corrupts the input data (noises up the network) by canceling out certain input values sporadically.

So, we simply set the values of a few input dimensions to 0. This forces the network to "denoise" the corrupted inputs. Hence the reconstructed data is similar to the original inputs but the model does not overfit.

This is because DAEs learn from a corrupted (partially visible input) distribution as compared to merely memorizing the input down to a tee. This indicates successful data retrieval and reconstruction from the latent distribution.

> The loss function must compare the network's outputs only with the original uncorrupted inputs.

---

## Add a V to the AE — Variational Autoencoders (VAEs)

Autoencoders use fixed vectors to map attributes from the input image to the latent space. This cripples the network's ability to better learn, customize, or tweak certain input attributes.

For example, in our input image say we have a woman with blue hair highlights. A simple autoencoder maps a fixed vector (a single value) to the "blue hair" attribute. It can probably only turn the blue hair on or off. But this is not the best way to quantify a transient attribute such as hair color.

On the other hand, variational autoencoders map a **probability distribution** (a range of values) for better control over the network's latent representations as compared to one rigid latent vector in autoencoders for representing complex attributes.

Imagine we are comparing an on/off switch with a rotating dial that you can twist for better control. VAEs resemble the more controllable and flexible dial. This also helps take into consideration the nuances in our data.

The VAE consists of generative and inference models for the tasks of compressing and reconstructing data. Let *x* be our observed variable (or input) and *z* be our latent variable. Our goal is to **infer** the latent variable from our input. For that we need to determine the conditional probability distribution of the latent variable given the input i.e. we need to find *p_theta(z|x)*.

### Why variational inference?

Here, while computing the probability distribution *p_theta(z|x)* for the input, we use the Bayes rule to expand:

> **Equation A: p_theta(z|x) = { p_theta(x|z) . p_theta(z) } / { p_theta(x) }**

Here:

- *p_theta(z|x)* is the posterior
- *p_theta(x|z)* is the likelihood
- *p_theta(z)* is the prior
- *p_theta(x)* is the marginal

We need to compute the marginal *p_theta(x)* but this is arduous since it integrates over all the values of *z*:

> **p_theta(x) = integral of p_theta(x|z) . p_theta(z) dz**

This computation is intractable (no efficient solution exists) in higher dimensional spaces as *z* being a latent space can have numerous values and dimensions. It requires exponential time for solving which is not plausible. Thus we resort to variational inference.

### Variational inference

By making use of another known and tractable yet simple Gaussian distribution *q_phi(z|x)*, we approximate *p_theta(x|z)*. So we need to ensure that *p_theta(z|x)* is modeled from *q_phi(z|x)* by converting this into an optimization problem.

We can achieve this with the **Kullback-Leibler (KL) divergence** measure. KL divergence tells us how different the two probability distributions are. For a discrete random variable and two distributions A and B:

> **KL(A||B) = Sigma A(x_i) . log( A(x_i) / B(x_i) )**

> KL(A||B) is not equal to KL(B||A) — the forward and reverse KL divergence measures are not equal and it is asymmetric in nature.

The KL divergence measure is non-negative and if the 2 distributions are mostly equal throughout, it amounts to zero. So for our 2 distributions p and q, we get:

> **KL( q_phi(z|x) || p_theta(z|x) ) = Sigma q_phi(z|x) . log( q_phi(z|x) / p_theta(z|x) )**

### Loss function derivation

In terms of the expectation and after some logarithmic simplification, the RHS becomes:

> **Equation B: KL = E_z [ log(q_phi(z|x)) - log(P_theta(z|x)) ]**

In equation B, KL = KL( q_phi(z|x) || p_theta(z|x) ) and **E_z** signifies the expectation over z wherein z is being sampled from q_phi(z|x).

Substituting equation A in equation B, we get:

> KL = E_z [ log(q_phi(z|x)) - log( {p_theta(x|z) . p_theta(z)} / {p_theta(x)} ) ]

> KL = E_z [ log(q_phi(z|x)) - log(p_theta(x|z)) - log(p_theta(z)) + log(p_theta(x)) ]

Although we still don't know *p_theta(x)*, note that the expectation term only computes over *z* and doesn't involve *x*. So we can move the term *log(p_theta(x))* to the LHS instead.

> KL - log(p_theta(x)) = E_z [ log(q_phi(z|x)) - log(p_theta(x|z)) - log(p_theta(z)) ]

Taking a minus sign on both sides and rearranging:

> log(p_theta(x)) - KL = E_z[ log(p_theta(x|z)) ] - E_z [ log( q_phi(z|x) / p_theta(z) ) ]

Note that the second term on the RHS is another KL divergence measure. Hence:

> **Equation C: log(p_theta(x)) - KL = E_z[ log(p_theta(x|z)) ] - KL( q_phi(z|x) || p_theta(z) )**

To arrive at the final loss function of a VAE, we attach a negative sign to the RHS of equation C:

> **L(theta, phi) = -E_z[ log(p_theta(x|z)) ] + KL[ q_phi(z|x) || p_theta(z) ]**

In the final loss function:

1. The first term is the **reconstruction loss**.
2. The second term is considered a **regularizer** term.

### The Reparameterization trick

As we know, the expectation term *E_z* represents *z* being sampled from *q_phi(z|x)*. Unfortunately, the sampling of *z* is a stochastic process, so we are dealing with a random variable. Random variables don't allow us to calculate their derivatives and hence aren't backpropagation-friendly.

This is a problem because we can't train our network with random variables that can't be backpropagated. Hence we use the **reparameterization trick**: represent the random variable as a deterministic variable through a transformation function *g_phi()*.

> **z = g_phi(epsilon, x)** and **z = mu + sigma . epsilon**

Here, *epsilon* is an extra stochastic random variable that allows *z* to become learnable due to the presence of mean *mu* and variance *sigma* parameters, hence making it training-friendly.

With this, we can better comprehend what goes on behind the scenes in a VAE. What's left is to build this model using a deep learning framework, train it, and later use it for inference purposes.

> You can find my PyTorch-based VAE implementation [here](https://github.com/01pooja10/Variational-Autoencoder).

---

## Vector Quantized VAEs (VQ-VAEs)

A VQ-VAE employs vector quantized representations of latent spaces and its encoder outputs "discrete" codes to model and then compress the input data. Further, the prior distribution is learned and not fixed.

The authors of the [VQ-VAE paper](https://arxiv.org/pdf/1711.00937.pdf) decided to incorporate discrete representations as they more accurately resemble naturally available data such as language which is a sequence of characters and images that can be described by words. Through this method, drawbacks such as posterior collapse and exploding variance are alleviated.

The architecture uses a convolutional neural network-based (CNN) encoder and decoder. Additionally, there is a vector quantization block between the encoder and decoder which uses an embedding space with embedding vectors **e_i** in R where **i** belongs to **{1, 2, ... K}**.

### Forward propagation

- **Encoder**: Uses a CNN to encode the input data *(x)* and produces *z_e(x)* of size D (the embedding size).
- We then calculate the one-hot posterior *q(z|x)* distribution. Discrete latent variables *z* are found using nearest neighbor lookup to match with one of *k* embedding vectors from the embedding space.

> **k** is assigned according to the L2 norm calculated between the encoder's output and embedding vector. This helps ascertain the embedding element nearest to **z_e(x)** i.e. the least distance between them.

- The term *q(z|x)* is indexed using the codebook to extract embedding vectors where integer *k* is the common link connecting *z_e(x)* to *z_q(x)*.
- We find the embedding vector (represented as *e_k*) nearest to *z_e(x)* for ensuring dimensional uniformity and assign it to *z_q(x)*.
- Now, *z_q(x)* is passed on to the decoder network for image reconstruction.

### Backpropagation: problems and solution

The codebook vector is not differentiable — gradients can't be propagated back through argmin as it maps *k* to discrete integers (indices of the closest vectors). Thus we use a **straight-through estimation** mechanism that copies the gradients of *z_q(x)* into *z_e(x)* since both vectors have the same dimensions.

### Loss function

The final VQ-VAE training loss has 3 terms:

- **First term: log p(x|z_q(x))** — The reconstruction loss optimizes both the encoder and decoder networks.
- **Second term: ||sg[z_e(x)] - e||^2** — The Vector Quantization (VQ) loss or codebook objective. The L2 error moves the embedding vector *e_i* closer to the encoded outputs *z_e(x)*. This term helps learn and update the embedding space.
- **Third term: beta . ||z_e(x) - sg[e]||^2** — The commitment loss ensures that the encoder's outputs don't vacillate or grow too much and stay close to the embedding space.

Here, **sg** is the stop gradient operator which signifies the absence of any gradients for the term it encompasses and uses a non-upgradable constant.

> VQ-VAEs achieve commendable likelihood for generating 128x128 color images, speech, and video action sequences too!

---

## Conclusion

In this blog, we have explored, analyzed, and understood how autoencoders have been evolving with new additions and conceptual improvements that enhance their generative capabilities.

What's next? The third blog in this series will be all about Generative Adversarial Networks (GANs) — the concepts used to build them, some useful math to understand their working, and more. Stay tuned!

Until then, if you have any doubts/suggestions, or would simply like to chat, feel free to reach out to me via [LinkedIn](https://www.linkedin.com/in/pooja-ravi-9b88861b2/).

---

## References and resources

1. [Autoencoders super blog — Lilian Weng](https://lilianweng.github.io/posts/2018-08-12-vae/)
2. [AE vs VAE blog](https://towardsdatascience.com/difference-between-autoencoder-ae-and-variational-autoencoder-vae-ed7be1c038f2)
3. [Another VAE blog — Jeremy Jordan](https://www.jeremyjordan.me/variational-autoencoders/)
4. [VAE video](https://www.youtube.com/watch?v=c27SHdQr4lw)
5. [VAE maths — video playlist](https://www.youtube.com/playlist?list=PLdxQ7SoCLQANizknbIiHzL_hYjEaI-wUe)
6. [VAE original paper](https://arxiv.org/pdf/1312.6114v10.pdf)
7. [Variational inference video](https://www.youtube.com/watch?v=HxQ94L8n0vU&t=946s)
8. [Evidence Lower Bound (ELBO) video](https://www.youtube.com/watch?v=IXsA5Rpp25w)
9. [VQ-VAE video](https://www.youtube.com/watch?v=VZFVUrYcig0)
10. [VQ-VAE blog](https://shashank7-iitd.medium.com/understanding-vector-quantized-variational-autoencoders-vq-vae-323d710a888a)
