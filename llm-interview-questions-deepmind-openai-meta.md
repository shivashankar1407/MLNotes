# Top 24 LLM questions asked at DeepMind, OpenAI, Meta and more!

*Source: [LinkedIn](https://www.linkedin.com/pulse/top-24-llm-questions-asked-deepmind-openai-meta-more-karun-thankachan-xlble/)*

---

In 2025, we saw a huge number of job descriptions asking for experience in LLMs. The same requirement was reflected in interviews, with candidates being asked LLM-specific questions.

So we talked to candidates, got a list of questions that were asked at some of the top "AI companies", and condensed it down to the following 25 questions. These 25 questions are not only the most commonly asked, but what we believe you should be able to answer as someone looking to build and productionalize LLM models.

Let's dive right into it!

---

## How does rotary positional embeddings (RoPE) work? [Google]

Rotary positional embeddings, or RoPE, give a transformer a sense of position by rotating each token's query and key vectors by an angle that depends on the token's position. You can think of every pair of dimensions in the vector as forming a little 2D plane. RoPE rotates that plane by a small angle for position 1, a slightly larger angle for position 2, and so on. Farther positions get larger rotations.

Because rotation changes direction but preserves magnitude, the model can tell how far apart two tokens are by comparing the difference in their rotations. Two tokens close together have similar angles. Tokens far apart have noticeably different angles. This means RoPE naturally encodes relative distance, which is exactly the information attention needs.

**Follow-up Question: Why do they outperform absolute positional embeddings in LLMs?**

Absolute positional embeddings take a simpler but less flexible approach. They learn a fixed vector for position 1, position 2, position 3, etc. The model sees these as extra "tags" added to each token. This works on the training lengths, but it does not teach the model how positions relate to one another. So if you ask the model to handle longer sequences or new patterns of spacing, it often struggles.

RoPE outperforms absolute positional embeddings for two main reasons. First, because the rotation angles change smoothly with position, transformers learn relational structure more easily. Self-attention really cares about differences between positions, not the positions themselves. RoPE encodes those differences directly. Second, RoPE helps with long-context extrapolation. Since the rotation is generated mathematically rather than looked up from a fixed table, the model can extend the pattern beyond the positions it saw during training. That is why modern long-context LLMs rely on RoPE almost universally.

So the core idea is: RoPE encodes position by rotating attention vectors in a way that naturally carries relative distance information. Absolute embeddings only label positions. RoPE gives smoother generalization and handles long sequences better, which leads to stronger performance in practice.

---

## What's the Chinchilla scaling laws? [Google]

The Chinchilla scaling laws came from a DeepMind paper that looked at how to most efficiently train large language models. The key idea was surprisingly simple: most LLMs before 2022 were far too big for the amount of data they were trained on. They had lots of parameters but weren't given enough tokens to fully learn from.

Here's the core insight. For a fixed compute budget, there is an optimal balance between model size (number of parameters) and training data size (number of tokens). Before Chinchilla, the common belief was that "bigger is better." People kept increasing model size and training on roughly the same 300B tokens. Chinchilla showed that this approach wastes compute and leaves performance on the table.

The Chinchilla finding was that performance improves much more when you reduce the number of parameters and increase the number of training tokens. Roughly speaking, if you double the number of parameters, you should also double the number of tokens. Their optimal regime suggested training should use about 20 times more tokens than parameters. For example, instead of training a 175B-parameter model on 300B tokens (like GPT-3), you could train a much smaller model on trillions of tokens and get better accuracy at lower compute.

This changed how people think about LLM training. It shifted the focus from "make the model bigger" to "balance size and data so both scale together." It also pushed the community toward collecting and cleaning much larger datasets, because simply increasing parameters no longer gives the best return. Another consequence is efficiency: with Chinchilla-style scaling, you can get GPT-3-level performance using dramatically fewer parameters, which reduces memory costs and inference costs while improving quality.

So the high-level message is that Chinchilla scaling laws showed that the bottleneck wasn't model size but insufficient training data, and that the smartest way to use compute is to scale parameters and training tokens together.

---

## What's the difference between causal attention and bidirectional attention? [Google]

Causal attention and bidirectional attention differ in what each token is allowed to "look at" when forming its representation. The choice affects what the model can learn and what tasks it is suitable for.

In causal attention, a token can only attend to tokens at positions before it. It cannot see the future. This creates a strict left-to-right flow of information, so the model learns to predict the next word based only on past context. This setup is essential for generative language models like GPT. If the model were allowed to peek at future tokens, it would cheat during training and would not generalize properly at inference time. Causal attention makes the model behave like a writer: it only knows what has been written so far, and it must guess the next word based on that history.

In bidirectional attention, a token can attend to both past and future tokens. Every word in the sentence can look at the full context. This is what models like BERT use. Because each token sees its entire neighborhood, the model becomes very good at understanding sentence structure, resolving ambiguity, detecting relationships between words, and performing tasks like classification or question answering. Bidirectional attention lets the model behave like a reader: it has access to the whole sentence and can interpret meaning holistically.

**Follow-up Question: What's appropriate for LLMs?**

These two forms of attention are appropriate in different parts of an LLM system.

Causal attention is used when the model must generate text. Autoregressive models need this constraint to ensure that the output flows naturally and is conditioned only on what has come before. Any system that writes text token by token—chatbots, code generators, story generators—depends on causal attention.

Bidirectional attention is used when the model must understand text, not generate it. Tasks like sentiment classification, document ranking, named-entity recognition, or natural-language inference benefit from seeing the whole span at once. Bidirectional models excel at comprehension but cannot be used directly for left-to-right generation because they do not learn to predict without future context.

In short: causal attention is for predicting the next word and producing fluent language; bidirectional attention is for interpreting full context and understanding meaning. Modern pipelines often use both: a bidirectional model for encoding or retrieval, and a causal model for generating responses.

---

## Why do you use a KV cache? (in the context of LLMs) [Google]

A KV-cache speeds up autoregressive decoding by reusing past attention computations instead of recomputing them at every new token. The basic idea is simple: when a transformer generates text one token at a time, most of the work it does is repeated unnecessarily unless you cache the intermediate states.

The KV-cache solves this by storing the keys and values from previous steps. Once you generate token 1, you compute its keys and values and save them. When you generate token 2, you compute only token 2's keys and values, then append them to the cache. Now the attention mechanism can attend to the entire history by simply reading from the stored cache without recomputing anything.

So at step t, the model only computes keys and values for the new token and reuses the cached ones for the previous t−1 tokens. Attention becomes:

- queries come from the new token
- keys/values come from the cached past plus the newly added ones

This dramatically reduces computation. Instead of cost growing roughly with t² (reprocessing the whole prefix each time), it grows roughly linearly: one forward pass per new token. That is the difference between a model that struggles to generate long outputs and one that can decode hundreds of tokens per second.

Because transformers are expensive precisely because they must process long sequences. If you force them to repeatedly reprocess the entire prefix, autoregressive generation becomes painful. By caching each layer's past keys and values, the model does almost no redundant work. Each decoding step becomes "small and constant" rather than "large and ever-growing."

This is why all modern autoregressive LLMs rely on KV-caching. Without it, long-context generation would be far too slow in real-world applications like chat, code generation, or search augmentation.

---

## What causes instability when training transformers at scale? [DeepMind]

Training transformers at scale can be unstable because small numerical issues get amplified through many layers of attention and residual connections. Three areas cause most of the trouble: normalization, initialization, and learning-rate schedules.

Normalization matters because it controls how gradients flow through the network. Early transformers used post-LayerNorm, where normalization happens after each block. That setup tends to be unstable in very deep models because gradients pass through large unnormalized transformations and can explode. Modern models almost always use pre-LayerNorm or RMSNorm. Pre-LayerNorm normalizes inputs before each block, which stabilizes gradient flow and prevents early collapse. RMSNorm is a lighter normalization that avoids some numerical issues in large models. Both approaches make deep transformers far easier to train.

Initialization is another major source of instability. Transformers depend heavily on residual connections, and if the weights are even slightly too large at the start, activations will grow layer by layer until they blow up. Stable training usually requires scaled initialization and sometimes explicit residual scaling so the network starts in a regime where the residual path does not dominate too early. Methods like DeepNorm formalize this idea. The goal is to keep activations and gradients in a safe range as depth increases.

Learning-rate schedules also play a huge role. If you start with a large learning rate, the model can diverge almost instantly because the early gradients are noisy. This is why warmup is essential. You start with a very small learning rate and slowly ramp it up over the first few hundred or thousand steps. After warmup, you typically use a decay schedule like cosine decay to keep updates stable over long training runs. Without warmup, models often collapse early; without decay, they can become unstable late in training.

Precision issues also matter. Using FP16 can cause overflows in attention computations, especially in softmax. Many large models use bfloat16 because it has a wider dynamic range. Gradient scaling, stable softmax kernels, and gradient clipping also help avoid numerical instability.

Finally, optimizer choices can contribute to instability. AdamW is standard, but its parameters matter. A learning rate that is too high or a β2 that is too large can cause oscillation or slow divergence over time.

So transformer instability usually comes from gradients becoming poorly behaved when depth, width, or sequence length get large. You mitigate it with safer normalization (pre-LN or RMSNorm), careful initialization and residual scaling, conservative learning-rate schedules with warmup and decay, and numerically stable precision settings. These practices collectively make it possible to train very large models without collapse.

---

## What does mixture-of-experts (MoE) architecture do? [DeepMind]

Mixture-of-experts (MoE) architectures work by making the model big on paper but cheap to run in practice.

Here's the basic idea in simple terms. Instead of having one huge feed-forward block inside each transformer layer, an MoE layer contains many feed-forward "experts," each specialized in different kinds of patterns. A routing network (usually a tiny learned linear layer followed by a softmax or top-k selection) decides which experts should handle each token. If the model has, say, 64 experts, you might activate only 2 of them for each token. That means the model's total capacity is equivalent to 64 feed-forward networks, but the compute cost per token is only as large as running 2 of them.

This is where the savings come from. In a standard dense transformer, every token passes through every layer in full. In an MoE transformer, most of the parameters stay idle on any given forward pass. Only the experts selected for that token do any computation. So compute grows with the number of active experts, not with the total number of parameters. The model can scale to billions or even trillions of parameters while keeping inference cost closer to that of a much smaller model.

Why does this still work well? Because different kinds of tokens naturally benefit from different computations. Some experts learn syntax, others learn rare vocabulary patterns, others focus on reasoning or numeric operations. The routing network learns which experts are best for each token type. Over time, experts become specialized, and the model ends up with much richer representational capacity than a dense model of the same compute budget.

This architecture solves the "scaling plateau" problem. Instead of training an enormous dense transformer where both compute and parameter count grow together, MoE lets you grow capacity without proportional compute, which gives better performance-per-dollar. Many modern systems, including models from Google and DeepMind, rely on MoE layers to reach very large capacities without blowing up training and inference cost.

So in short, MoE architectures reduce compute by only activating a few expert networks per token. They maintain high capacity by having many experts available, each of which contributes specialized knowledge, even though most remain inactive at any moment. The result is a model that behaves like a huge transformer but runs like a much smaller one.

---

## Can you walk me through a typical RLHF pipeline? [OpenAI]

The RLHF pipeline has three major steps: supervised fine-tuning, reward modeling, and reinforcement learning. Each step corrects a different limitation of the pretrained model.

**Supervised fine-tuning (SFT)** is the warm start. You take a pretrained LLM and fine-tune it on a curated dataset of prompts paired with high-quality human responses. The objective is standard supervised learning: maximize the likelihood of the human-written output given the input. Technically, this is cross-entropy training on next-token prediction, just like pretraining but restricted to good answers. This gives the model a helpful baseline and stabilizes its behavior. Without SFT, RL tends to push the model into unstable or degenerate regions of parameter space.

**Reward modeling (RM)** teaches the system what "good" means. Humans are shown a prompt and two or more candidate responses generated by the SFT model. They pick which one they prefer. These preference pairs are converted into a training signal for a reward model. The reward model typically shares a backbone with the base LLM but adds a scalar head that produces a single score for each response. The model is trained using a pairwise loss such as the Bradley–Terry or logistic preference loss. The idea is that the reward model should assign a higher reward to the response humans chose and a lower reward to the one they rejected. After training, the RM acts as a learned proxy for human judgment and can evaluate millions of responses cheaply.

**Reinforcement learning (RL)** then tries to optimize the LLM to generate high-reward responses. The most common method is PPO (Proximal Policy Optimization). During PPO training, the model generates responses to prompts. Those responses get scored by the reward model. PPO then updates the LLM's parameters to increase the probability of responses with higher rewards. At the same time, PPO adds a KL-divergence penalty that keeps the updated model close to the SFT model. This prevents reward hacking, where the model learns shortcuts that exploit the reward model rather than producing genuinely helpful responses. In practice, the PPO objective balances three things: reward maximization, policy stability, and proximity to the SFT policy.

With repeated cycles of generate → score → update, the model learns to produce responses that humans consistently prefer. Unlike SFT, which only teaches imitation, RLHF can adjust the model's deeper behavior, such as reducing harmful outputs, improving helpfulness, or aligning with conversational norms.

---

## How does preference modeling differ from reward modeling? [OpenAI]

Preference modeling and reward modeling are closely related, but they solve two slightly different problems and are used in different parts of the alignment pipeline. The clearest way to explain the difference is this:

**Preference modeling** learns how humans compare responses. **Reward modeling** learns how to score responses.

They often use the same data, but the output and use cases differ.

Preference modeling works directly with human comparisons. You give a human a prompt, show them two or more responses, and ask which one they prefer. The model then learns a function that predicts these pairwise preferences. It does not try to assign an absolute numerical score. Instead, it learns which answer is better for a given pair. This is useful when humans are bad at giving absolute ratings like "this is a 7 out of 10" but are very good at choosing between two alternatives. Preference modeling is also naturally robust because it focuses on relative quality: "A is better than B," not "A is 0.62." You use preference models when you care about ranking quality, comparing candidate responses, or creating pairwise datasets for training downstream components.

Reward modeling converts those comparisons into a function that outputs a single scalar value for any response. Humans still provide pairwise preferences, but the model is trained to produce a numerical reward such that higher values correspond to more preferred outputs. This scalar reward allows us to plug the model into reinforcement learning algorithms like PPO, where the model optimizes expected reward. In other words, reward modeling is preference modeling transformed into something RL can use. You use reward models when you need an automatic, fast scoring function that can evaluate millions of samples during RL training.

So preference modeling is about learning relative human judgments, while reward modeling is about converting that relational knowledge into an absolute scoring system. You would use preference modeling when collecting data or when your downstream tasks require ranking. You would use reward modeling when training the generative model with RL, because only the reward model produces the scalar signal needed for optimization.

---

## Why does hallucination happen, and how do you tackle it? [OpenAI]

Hallucination is when an LLM gives an answer that is fluent and confident but factually wrong or completely made-up. It happens because the model is trained to produce likely text, not to check truth. During training it learns statistical patterns from huge datasets, so if a question pushes it into an area where it has weak signal, missing knowledge, or ambiguous context, it simply generates something that sounds plausible. In other words, the model is optimizing for coherence, not accuracy.

Several things cause hallucinations. One is the model's tendency to "complete a pattern" even when the input doesn't support it. If the model has learned that biographies usually include birthplaces or dates, it may invent them if it is unsure. Another cause is misalignment between user intent and the training objective. The model does not have an internal fact-checking mechanism or memory of verified truths. It only knows what words tend to follow other words. There is also exposure bias: during training the model never sees its own mistaken outputs, so it doesn't learn to correct them. And when prompts are vague or missing key details, the model may fill in gaps using patterns from training data rather than sticking to what is known.

There are a few practical techniques that reduce hallucinations at inference time. One is constraining generation: using retrieval-augmented generation (RAG), where the model retrieves relevant documents and grounds itself in that evidence. When the model is tied to real context, the chances of free-form invention drop. Another is temperature control. Lower temperature makes the output more deterministic and less likely to wander into speculative completions. Models hallucinate more when sampling is wide and creative.

You can also use chain-of-thought verification approaches: ask the model to reason step-by-step and then to check its own reasoning, or run multiple sampled answers and choose the one that is most consistent across samples ("self-consistency"). In some systems, you run a second model pass where the model critiques or fact-checks its first answer. This doesn't eliminate hallucinations but reduces obvious ones.

Prompting also matters. Direct, constrained questions like "quote from the passage above" lead to fewer hallucinations than open-ended requests. Asking the model to say "I don't know if I'm not sure" lowers the incentive to invent answers; many modern alignment techniques push the model to do that more reliably.

Finally, if the domain is sensitive (legal, financial, medical), you can put a guardrail model or a rule-based checker in front of the output. The model's generation is still probabilistic, but guardrails catch fact mismatches, unsupported claims, or format violations.

---

## What's the point of tools and function calling with LLMs? [Amazon]

Tools and function calling give an LLM something it does not naturally have: the ability to act, look things up, and use external systems instead of guessing. Without tools, an LLM can only generate text based on patterns from training data. With tools, it can interact with the world.

Here's the core idea in simple terms.

An LLM is great at language but bad at precise tasks like math, retrieval, code execution, database queries, or interacting with APIs. When it doesn't know something, it tends to hallucinate. Tools and function calling fix this by letting the model decide when it needs outside help, describe the task in a structured format, and let an external system actually perform the computation. The LLM then uses the returned result to produce a final answer.

So instead of the model guessing an answer to "What's the weather in Paris right now?" it calls a weather API. Instead of hallucinating the contents of a company database, it issues a structured query. Instead of bluffing through a complicated calculation, it offloads the math to a calculator. The model becomes more like a planner that delegates specialized tasks to the right tool, then integrates the results into a coherent response.

The benefits are huge. Function calling dramatically reduces hallucinations because the model no longer has to fabricate facts. It improves reliability in domains like finance, travel, medicine, or code generation where precision matters. It also makes LLMs much more practical in real products, because they can interface with existing systems and workflows instead of staying confined to text generation.

The point of tools and function calling, in short, is to turn an LLM from a very smart text generator into a useful agent that can retrieve real information, execute exact operations, and interact with real-world systems safely and predictably.

---

## If an LLM keeps producing excessively verbose answers, how would you correct it? [OpenAI]

If an LLM keeps giving overly long answers, I first treat it as a debugging problem: figure out why the model thinks it should be verbose, then fix the easiest causes before touching the training pipeline.

The first thing I check is whether the problem comes from inference settings. High temperature, no stop sequences, a large max-tokens limit, or a system prompt that asks for "thorough explanations" will all push the model to ramble. I also check if the examples in the system prompt or few-shot prompts are long; the model will imitate whatever it sees.

If inference looks fine, I look at the training data. Sometimes the supervised fine-tuning dataset contains mostly long, detailed answers, so the model learns that "good answers = long answers." Or the reward data used to train the reward model might accidentally prefer longer replies, so the reinforcement learning step pushes the model toward being exhaustive.

To diagnose this, I try simple prompts like "answer briefly in one sentence." If the model still refuses to be concise, it usually means the alignment process over-rewarded long responses. That's a sign the issue is inside the training pipeline, not just the prompting.

The easiest fixes start at inference: lower temperature, set a shorter max token limit, add a direct instruction like "respond concisely," and include one or two short examples in the system prompt. Those alone often solve the issue.

If not, the next step is to adjust the reward or training signals. You can add human preference data where people explicitly choose shorter answers over unnecessarily long ones. This retrains the reward model to "see" conciseness as good, instead of thinking that detailed answers always win. You can also apply a small length penalty during reinforcement learning or strengthen the KL penalty so the model stays closer to the supervised fine-tuned behavior.

Another simple option is to fine-tune the model on a small set of concise, high-quality answers. A short fine-tuning run can correct the bias toward verbosity without a full retraining cycle.

Finally, from a user-experience standpoint, it sometimes helps to reply briefly by default and offer something like "I can explain in more detail if you want." That pattern keeps answers concise while still giving users access to depth when they need it.

This is a bit of an open-ended question, so expect follow-ups.

---

## What's the difference between pretraining loss, SFT loss, and RLHF reward? [OpenAI]

Pretraining loss, SFT loss, and RLHF reward all shape an LLM, but they push it in different directions because they optimize different objectives. The easiest way to understand the difference is to look at what each stage is trying to teach the model.

**Pretraining loss** is about learning general language patterns. During pretraining, the model sees huge amounts of raw text and tries to predict the next token. The objective is pure next-token prediction: minimize cross-entropy between what the model predicts and the actual next token. This trains the model to understand grammar, facts in the data, reasoning patterns, and how language works in general. But it does not teach the model to follow instructions or behave helpfully. It just learns to continue text in a statistically plausible way.

**Supervised fine-tuning (SFT) loss** is about learning how humans want the model to answer specific kinds of prompts. In SFT, the model is shown prompts paired with high-quality human-written answers. It still uses cross-entropy, but now the target text is "good assistant behavior," not random internet text. This makes the model more helpful, less chaotic, and better aligned with instructions. The key difference from pretraining is that SFT is behavior-shaping rather than general language learning.

**RLHF reward** is not a loss in the same sense. It is a scalar score produced by a reward model that tries to reflect human preference. The RL objective is to maximize expected reward, not to mimic text. During RLHF, the model generates answers, the reward model scores them, and an algorithm like PPO adjusts the model to produce higher-scoring answers. This step teaches high-level preferences such as politeness, conciseness, safety, reduced hallucination, tone, and helpfulness. The RL objective does not care about matching a specific target sequence; it cares about producing better sequences according to human judgments.

So they differ because each stage is solving a different problem. Pretraining teaches broad linguistic knowledge. SFT teaches explicit instruction-following and example-based behavior. RLHF teaches preference optimization, where the model learns to choose responses humans would prefer rather than responses humans wrote.

You need all three because language modeling alone does not make a good assistant, imitation alone does not capture nuanced preferences, and preference optimization alone is too brittle without the stable foundation that pretraining and SFT provide.

---

## What's Constitutional AI? [Anthropic]

Constitutional AI is an approach to aligning LLMs that tries to reduce the amount of direct human labeling and preference comparison needed in RLHF by giving the model a written "constitution" of rules. Instead of humans repeatedly judging outputs, the model critiques and revises its own answers using those rules. In other words, it replaces most human preference data with model-guided self-improvement, constrained by a set of safety and behavior principles.

Just to crystallize the idea, let me explain how it works with RLHF.

Constitutional AI begins with a constitution: a list of high-level principles that describe safe and desirable behavior. These can include things like avoiding harmful advice, staying nonjudgmental, being truthful, maintaining privacy, offering alternatives instead of refusals, and so on. They act as the rules the model should follow when deciding how to correct or critique an answer.

The pipeline then has two main phases. First is **self-critique and revision**. You prompt the model with a question and let it produce an initial answer. Then you ask the model to evaluate its own answer by referencing a specific principle from the constitution. The model critiques the answer and suggests improvements such as "this response may be unsafe because…" or "according to principle X, the answer should avoid giving medical instructions." After critiquing, the model produces a revised answer that better follows the principle. You can run this process across many prompts to generate a dataset of improved answers. This dataset is used for supervised fine-tuning. It's similar to SFT in RLHF, but instead of humans writing the improved answers, the model produces them under constitutional guidance.

The second phase is **preference training**, but instead of humans picking between responses, the model generates pairwise preferences based on the constitution. It compares two answers and decides which one better follows a given principle. These comparisons train a preference model or are used directly for loss functions like DPO (Direct Preference Optimization). This step replaces the human preference-pair labeling that RLHF normally relies on. The idea is that the constitution gives a consistent standard, so the model can generate a lot of alignment data cheaply.

This changes the RLHF pipeline in a few ways. It removes most of the human-in-the-loop labeling, because the model critiques and ranks its own answers using the constitution. It also avoids the reinforcement-learning instability of PPO by pairing the revised answers with simpler, supervised training objectives or direct preference optimization. The alignment signal becomes more consistent because the "judge" is guided by fixed principles rather than subjective human raters. And because the alignment is generated at scale, you can shape many behaviors—like politeness, harmlessness, grounded refusals—without collecting huge preference datasets.

So the shift is essentially this: instead of humans telling the model which responses are better, the model uses a set of written principles to judge and improve its own outputs.

---

## How would you design a system that constrains an LLM to be helpful without revealing sensitive internal knowledge? [Anthropic]

First, you define what "sensitive internal knowledge" means for your company. That might include source code, internal metrics, customer data, unreleased product details, passwords, API keys, or anything that should never appear in an LLM response. Having clear definitions makes it easier to build guardrails around them.

Next, you make sure the model is not trained on anything that contains secrets. Many leaks happen because sensitive text accidentally ends up in the training corpus. So you filter or redact internal documents before they are ever used for pretraining or fine-tuning. If you really need internal knowledge for the model to do its job, you use summarized, sanitized versions instead of the raw content.

When you use retrieval (RAG), you only let the LLM access approved documents. The retrieval layer should enforce permissions, check the user's role, and redact sensitive fields before anything reaches the model. The model never sees raw secrets; it only sees cleaned, authorized content. This prevents it from accidentally spitting out internal details.

The system prompt also matters. You explicitly tell the model not to reveal confidential information and to politely refuse when asked. You show it examples of safe refusals so it learns the behavior you expect.

At runtime, you add detectors that check both the input and the output for signs of sensitive content. These can include PII detectors, secret-pattern regexes, or small classifiers trained to spot unsafe disclosures. If the model ever tries to output something suspicious, you block or rewrite the response.

If the LLM has the ability to call tools or APIs, you never let the model call them directly. A policy layer in the backend checks whether the user is authorized and whether the arguments are safe. This prevents the model from accidentally leaking data by asking a tool to fetch something that should be private.

You can push alignment further by fine-tuning the model on examples where the correct response is refusal. You can teach the reward model to penalize outputs that contain confidential content. Over time, the model becomes naturally cautious about revealing anything sensitive.

For situations with very high risk, you route certain queries to human reviewers before returning the answer. This human-in-the-loop step acts as a final safety net for things like legal, medical, or financial information.

Finally, you continuously monitor the system. You log prompts and responses in a privacy-safe way, watch for spikes in blocked content or user complaints, and run regular red-team tests to see whether the model can be tricked into leaking information. When you find weak spots, you tighten the filters or update the training data.

So the big picture is this: remove sensitive data from training, restrict what the model can see at inference, guide it with clear instructions, check everything it outputs, and enforce permissions on any actions it takes. Layering all of these protections together keeps the model helpful while making unintended leaks extremely unlikely.

---

## Why would you use Chain-of-Thought reasoning? [Anthropic]

Chain-of-thought (CoT) reasoning helps an LLM explain its intermediate steps instead of jumping straight to an answer. This usually improves correctness, because the model slows down, breaks the problem into smaller pieces, and uses more deliberate reasoning patterns. When models generate their steps, they tend to catch arithmetic mistakes, clarify assumptions, and follow logical structure more reliably. That is why CoT often boosts accuracy on math, logic puzzles, or complex analysis.

**Follow-up Question: Any downsides to CoT?**

CoT increases the risk of revealing unsafe or sensitive reasoning. When the model exposes its intermediate thoughts, it may accidentally reveal details that were never meant to be shown. For example, in safety-sensitive domains like security, harmful instructions, or confidential reasoning, chain-of-thought can reveal methods, decision paths, or internal heuristics that should remain hidden. Even if the final answer is harmless ("I cannot help with that"), the step-by-step reasoning might describe the very thing you don't want disclosed.

There are a few ways this happens in practice. One is that the model tries to reason through an unsafe request before refusing, and the reasoning itself contains the harmful information. Another is that the model uses CoT to justify why something is unsafe, but ends up describing the dangerous action in the process. CoT also increases the surface area for hallucination: the model may invent plausible-sounding intermediate steps that are inaccurate or misleading, which can be especially harmful in medical, legal, or technical contexts.

**Follow-up: How would you solve for it?**

Production systems often separate internal reasoning from public output. They allow the model to generate hidden chain-of-thought internally but only show the final answer or a short justification. This keeps the correctness benefits (the model still thinks step by step) while reducing the risk of exposing raw reasoning. Some systems also train the model to produce concise, safe rationales instead of full CoT, or they use a verifier model to check whether the reasoning contains unsafe material before anything is shown to the user.

---

## What does distillation do? [Meta]

Distillation is a way of training a smaller student model to imitate the behavior of a larger teacher model so that the smaller model keeps most of the teacher's performance while being much faster and cheaper to run. In other words, the big LLM "tutors" the small SLM until the SLM learns to behave almost the same way, but with far fewer parameters.

Just to break it down a bit more:

First, you take a strong, fully trained LLM and let it generate high-quality outputs on a large set of prompts. These prompts may come from public datasets, synthetic data generated by the teacher itself, or logs from real usage (with privacy controls). The teacher model produces answers, rationales, probabilities over tokens, or other signals that capture how it reasons.

The smaller student model is then trained to match the teacher's behavior. Instead of training on human-written text, the student optimizes a loss that penalizes any difference between its predictions and the teacher's. The most common signal is soft targets: the full probability distribution the teacher assigns to the next token. These distributions contain far more information than a single "correct" answer because they show which alternatives the teacher thinks are somewhat plausible. Learning from these soft probabilities helps the student absorb the teacher's knowledge more efficiently.

As the student trains, it becomes better at approximating the teacher. Even though the student has fewer layers or a smaller hidden size, it starts reproducing the teacher's behavior with surprisingly high fidelity. In practice, good distillation can preserve most of the accuracy on downstream tasks while cutting inference cost dramatically.

---

## Why does SLMs sometimes outperform LLMs in retrieval-augmented systems? [Meta]

Smaller SLMs can sometimes beat much larger LLMs in retrieval-augmented systems because, in RAG, the hard work is done by the retriever, not the generator. The model's job is often to read the retrieved passages and answer accurately, not to invent answers from its own internal knowledge. That shifts the advantage toward models that are precise, grounded, and obedient, rather than models that rely heavily on their own parametric memory.

Also, smaller models are often easier to control with prompting. In a RAG setup, you want the model to "stick to the provided evidence." Smaller models tend to follow these instructions more consistently. Larger models, because they are more expressive and creative, sometimes drift or over-explain, which can reduce factual accuracy.

It can also be because smaller models are less prone to overfitting spurious patterns. Large LLMs sometimes over-interpret retrieval snippets or infer connections that are not actually present. An SLM is more literal. It extracts what is written rather than speculating. In practical retrieval tasks, being literal can improve correctness.

---

## How do you approach building a RAG Pipeline? [Amazon]

To design a retrieval augmented generation pipeline for an enterprise knowledge base, I would start by making sure the underlying content is clean and well organized. This means collecting trusted internal documents such as product guides, internal wikis, support documentation, and FAQs, then converting them into consistent text. I would split long documents into smaller chunks that are a few hundred tokens in length. Each chunk would carry metadata such as where it came from, when it was updated, and what access permissions apply. Good preprocessing at this stage makes retrieval more reliable later.

The next piece is indexing. I would build both a dense embedding index and a sparse keyword index like BM25. Dense retrieval helps with natural language questions such as someone asking how to perform a task. Sparse retrieval helps with exact matches such as error codes or product names. When a user sends a query, the system uses both methods and merges the results. It then applies a reranking model that examines the top candidates in more detail and chooses the best ones to pass to the language model.

Once the system has the right evidence, it constructs a prompt that contains the user question and the selected chunks of text, along with clear instructions. The instructions tell the model to answer only using the provided information, to cite the specific documents that support the answer, and to decline politely if the retrieved text does not contain enough evidence. This is important because in an enterprise setting accuracy matters more than creativity. The model should behave like a grounded summarizer that draws from the retrieved documents instead of relying on its own stored knowledge.

For higher risk uses, I would add a verification step. A small checker model can confirm whether the important claims in the output actually appear in the retrieved documents. If the checker finds inconsistencies, the system can retry retrieval, ask clarifying questions, or route the case for human review. This step helps prevent hallucinations and keeps answers trustworthy.

Access control is also essential. Before any retrieval happens, the system should check the user's permissions. The retriever must only search collections the user is allowed to view. Sensitive content such as personally identifiable information should be redacted at ingestion time and again before text is shown to the model. This prevents accidental leakage in the final answer.

I would also design the pipeline so it updates easily. When a document changes, only the affected chunks need to be reembedded rather than rebuilding the entire index. Monitoring is important too. I would track retrieval quality, helpfulness of the final answers, latency, error reports, and any cases where the model produced an answer that was not grounded in the documents.

In domains such as legal, medical, or finance, a human review path can make the system safer. If a query is sensitive, the model can still propose an answer but a human checks it before it is returned. The model can assist by highlighting the most relevant passages, which speeds up the review.

In summary, a solid enterprise RAG pipeline has a few essential qualities. It has clean and well structured documents. It uses a combination of dense and sparse retrieval with reranking. It constructs prompts that keep the model grounded. It enforces permissions and privacy. It adds verification for safety. And it updates and monitors the system so it stays accurate over time. The language model becomes a reliable synthesizer of the company's knowledge rather than a source of unverified information.

---

## How would you fine-tune an LLM to support company-specific jargon while minimizing catastrophic forgetting. [Amazon]

If you want an LLM to understand and use company-specific jargon without forgetting everything else it knows, the main goal is to add new knowledge without overwriting the base model. Catastrophic forgetting happens when you fine-tune too narrowly and the model starts drifting away from its general language abilities. So the strategy is to inject new information gently and protect the original behavior.

A practical first step is to avoid full fine-tuning of all weights. Instead, use lightweight methods like LoRA or QLoRA adapters. These add small trainable layers on top of the base model while keeping most of the model frozen. Because the backbone doesn't change, the model retains its broad knowledge, and the new adapters learn the company jargon. This alone already reduces forgetting significantly.

Next, build a clean dataset of examples that show how the jargon is used. These could be internal product descriptions, customer FAQs, internal terminology lists, or realistic Q&A pairs. Include a bit of general text in the mix too, so the model still sees normal language while learning the new terms. This "mixed training" reminds the model not to forget its old skills.

During training, keep the learning rate small and the number of training steps modest. You want the model to nudge in the direction of your jargon, not overhaul itself. If you really want to be safe, you can freeze the entire model except the adapters. You can also use small regularization tricks that push the model to stay close to its original predictions.

Sometimes jargon changes quickly or is too detailed for training. In that case, it's better to use retrieval: store the official definitions of terms in a database and let the model look them up when needed. This avoids forgetting entirely because you aren't changing the model's weights—you're just giving it extra context at inference time.

After training, you should test both the new jargon skills and the old general abilities. Make sure the model still writes good answers about everyday topics, not just internal ones. If you see the quality slipping, you can adjust the ratio of jargon data to general data or lower the learning rate.

The final step is to deploy gradually. Introduce the updated model to a small slice of traffic and monitor whether its general responses, tone, and safety behavior remain stable. If everything looks good, roll it out more broadly.

---

## Why LLMs hallucinate even when retrieval systems are attached? [Amazon]

LLMs can still hallucinate even when you attach a retrieval system, because retrieval alone does not force the model to use the evidence it receives. An LLM is trained to produce fluent text based on statistical patterns. When it gets external context, the model often treats that context as just another suggestion rather than a hard constraint. As a result, it may blend retrieved facts with its own internal memories, fill gaps with guesses, or ignore the retrieved text entirely if it believes it "knows better."

There are a few common failure modes. Sometimes the retrieved documents do not exactly answer the question, so the model improvises to fill the holes. Sometimes the retrieved context contradicts what the model learned during pretraining, and the model chooses its own beliefs over the provided evidence. And often the prompt is not explicit enough about using the retrieved text, so the model slips into general language generation based on its internal priors.

**Follow-Up Question: How can you enforce grounding?**

To enforce grounding, you need mechanisms that make the model rely on retrieved evidence rather than its internal assumptions. One simple method is to write very clear instructions that say the model must answer only based on the provided context and must not invent details. Few-shot examples help too: show the model an example where it cites the source for each claim. If the prompt includes language like "cite your sources" or "only use the context provided," the model is more likely to stay faithful to the retrieval results.

Another approach is training. You can fine-tune the model on datasets where the outputs are explicitly grounded in retrieved context. The training signal penalizes the model if it generates claims not present in the retrieved documents. This teaches the model that its job is to synthesize from the evidence, not to draw on its own knowledge. Some teams also use a verifier or reranker that checks whether the final output is actually supported by the retrieved content—if not, they discard and regenerate.

Finally, you can use structural constraints. Some systems format the output as extracted facts plus citations, rather than letting the model produce free-form text. By requiring citations, you force the model to stay close to the source material. This is common in production RAG systems where accuracy is critical.