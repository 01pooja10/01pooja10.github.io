As an active researcher and developer in the field of Artificial Intelligence, I decided to document my learning journey through blogs since they are effective in imparting knowledge and furthering curiosity for both you and me. This blog will serve as a succinct preamble to a comprehensive series that will be published in the upcoming months.

Through this series, I aim to elucidate the working and further touch upon the mathematics behind various state-of-the-art generative models. It will include models such as VAEs, GANs, normalizing flows, stable diffusion, and GPTs. Since this is the first blog, let's get acquainted with generative AI and its potential in today's technology-dominated atmosphere.

---

## Generative AI — the what and the why

**Generative AI** (or GenAI) is a branch of Artificial Intelligence that uses generative models. It employs deep learning-based algorithms to essentially *generate* or produce new and hopefully, relevant data corresponding to the inputs provided.

This phenomenon was initially (sort of) kindled by Hidden Markov Models (HMMs) which predicted the next data sequence given the current state. Back in the day, HMMs were notably used in speech recognition and time series analysis. Nowadays GenAI is used for generating not just text but images, videos, code snippets, and synthetic datasets.

With the public release of [ChatGPT](https://openai.com/blog/chatgpt) which uses Generative Pre-Trained Transformers (GPTs), a family of generative models from OpenAI, the company itself and its CEO Sam Altman have become frontrunners in the GenAI race. ChatGPT also dragged into the spotlight how efficaciously GenAI tools can be fully leveraged by laypersons and academicians alike. Now millions of users arrive at quick solutions to myriad problems by merely dropping their queries as prompts in the ChatGPT interface.

ChatGPT garnered over 100 million users only a mere two months after its launch. It has turned heads across the world and also attracted billions of dollars worth of investments from tech giants such as Microsoft which integrated ChatGPT into its Bing search engine. Further, [Github Copilot X](https://github.com/features/preview/copilot-x) has adopted OpenAI's GPT-4 model to power its AI-based code completion feature.

Further, Generative AI's widespread usage can be seen in the following applications recently released by other major tech giants:

1. [Adobe Firefly](https://www.adobe.com/sensei/generative-ai/firefly.html) uses the image generation capability of AI to assist in editing, extending, projecting, or even transforming images.
2. [Stable diffusion](https://stability.ai/blog/stable-diffusion-public-release) models from stability.ai have been used to generate various photo-realistic images of text prompts provided by users based on their requirements.
3. [Midjourney AI](https://www.midjourney.com/home/?callbackUrl=%2Fapp%2F) has a dedicated discord server with text prompt-based image generation capabilities.
4. NVIDIA's [Avatar Cloud Engine](https://www.nvidia.com/en-in/geforce/news/nvidia-ace-for-games-generative-ai-npcs/) (ACE) is a tool that uses Generative AI for instilling behavioral/personality traits in virtual characters.
5. Google's [MusicLM](https://google-research.github.io/seanet/musiclm/examples/) and Meta's [MusicGen](https://huggingface.co/spaces/facebook/MusicGen) are both text-to-music generative models that can produce musical sequences from text prompts.

---

## Concerns regarding Generative AI

Although generative AI is becoming ubiquitously useful in accelerating productivity, some still perceive it as a threat. There has arisen speculation and criticism regarding the following aspects of utilizing GenAI in our day-to-day lives:

1. **Ethical aspects of building such software** — Certain jailbreak techniques or prompts help circumvent the legal boundaries of generative models, forcing them to generate immoral or unethical answers. This imperils the usage of such models and exposes their vulnerabilities.

2. **Transparency of decision-making** — How generative models arrive at the end result is at times a gray area for the layman given their black-box architectures and arcane training processes. Ideally, these models' outputs and suggestions shouldn't be taken at face value while making important decisions.

3. **The origin of training data** — Where the training data is sourced from remains a concern since publicly available text or image data is not immune to human bias and may hence be skewed in its representation or less in fairness quotient.

4. **Job security for human workers** — Another rising point of contention is how safe are our jobs from AI. Some believe GenAI's exceptional capability to produce new/original content will eventually supplant human workers from their traditional work roles so that corporations can boost productivity and produce better results.

5. **How reliable (or biased) are the outputs** — Significant research must be pursued to determine what kind of bias patterns are frequently noticeable in the outputs generated by the AI software. Pinpointing where the model is skewed by human bias and how/if it reinforces any stereotypes through its outputs, is imperative to building equitable systems.

6. **How open-sourced each model is** — Huge corporations and tech giants should advocate for transparency in coding, training, testing, and releasing different versions of their generative models so as to healthily promote research and development (R&D) in such sectors and improve accessibility.

Bearing all this in mind, we must come to realize: whether such criticisms hold true or not and to what extent they do — is not going to change the fact that research, as well as public interest in these areas, is raging.

> When generative models are democratized for public use through accessible user interfaces, they gain astronomical popularity.

Hence, if you are interested in either jumping on the bandwagon or just brushing up on the latest developments, this blog series is your best friend.

---

## The all-in-one timeline for generative models

Let's check out how generative models have been evolving! There is a supremely informative [chart on Kaggle](https://www.kaggle.com/discussions/getting-started/397345) which tracks the release of various generative models and the category they belong to.

As the graph depicts, we can see the rapid advancement of Generative AI with the advent of variational autoencoders (VAEs) in 2013 which can be considered an improvement over the preexisting autoencoders but with variational inference. These concepts are thoroughly explained in the [next blog on Autoencoders](post.html?post=autoencoders).

Further, Generative Adversarial Networks (GANs) were brought into the limelight by Goodfellow et al. in 2015 and with this, we saw a flurry of models being released: DCGAN, GANs for image-to-image translation, Wasserstein GAN, StyleGAN, and many more. I plan on dedicating one blog in this series to exploring GAN-related concepts and analyzing some popular GANs along with their code implementations.

For an example of what GANs are capable of, here's [HyperStyle](https://arxiv.org/abs/2111.15666) — a modification or variation of StyleGAN from Alaluf et al. — it can modify/edit a person's features such as facial hair, hairstyle, age, etc.

In the language modeling sphere; RNNs and LSTMs ceased to remain the only options for language-oriented tasks. The [Attention is all you need](https://arxiv.org/abs/1706.03762) paper by Vaswani et al. released in 2017 paradigmatically shifted the focal point of NLP research from simple recurrent networks to transformers and LLMs. Such breakthroughs have paved the way for today's powerful generative beasts. Now, companies are racing against time to involve themselves and outdo their competition in leveraging the generative capabilities of Artificial Intelligence.

Lately, we have been witnessing the power of GenAI-based research such as OpenAI's DALL-E 2/ChatGPT, stability ai's stable diffusion/ControlNet, and Google's Bard powered by LaMDA. These are just a few of the many widely discussed models that show how powerful Generative AI can be when implemented astutely.

I will dip all our toes into the vast ocean of Generative AI with this blog series. I also plan on including links to my own GitHub code repositories of select generative models so that interested readers can learn the implementational details. With that said, I plan on releasing separate blogs for explaining the following types of generative models:

1. Autoencoders
2. GANs
3. Transformers/Language models
4. Flow-based generative models
5. Diffusion models

---

## Conclusion

This blog series not only incentivizes me to compactly present/document all that I know about these topics but also thoroughly learn the concepts before penning them down for my readers (you) to enjoy. So I am very excited to see this journey through patiently. In the [next blog](post.html?post=autoencoders), we will dive into the working mechanism behind autoencoders, denoising autoencoders, variational autoencoders, etc. I will catch you all there!

Until then, if you have any doubts/suggestions, or would simply like to chat, feel free to reach out to me via [LinkedIn](https://www.linkedin.com/in/pooja-ravi-9b88861b2/).

---

## References and resources

1. [Intuitive video on Hidden Markov Models (HMMs)](https://www.youtube.com/watch?v=RWkHJnFj5rY)
2. [Detailed blog on HMMs](https://jonathan-hui.medium.com/machine-learning-hidden-markov-model-hmm-31660d217a61)
3. [Learn more about ChatGPT — video](https://www.youtube.com/watch?v=0A8ljAkdFtg)
4. [Try out ChatGPT](https://chat.openai.com/)
5. [Try out ControlNet](https://stablediffusionweb.com/ControlNet) or [Stable Diffusion](https://stablediffusionweb.com/#demo)
6. [Research paper: HyperStyle](https://arxiv.org/abs/2111.15666)
7. [Research paper: Attention is all you need](https://arxiv.org/abs/1706.03762)
