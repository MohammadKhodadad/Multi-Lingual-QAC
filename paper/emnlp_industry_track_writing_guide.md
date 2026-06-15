### Source: https://aclrollingreview.org/responsibleNLPresearch/

# EMNLP Industry Track Paper Writing Guide

This guide focuses on the **idea and writing aspect** of an EMNLP Industry Track paper, assuming you already have the official ACL/EMNLP template.

## Core Writing Principle

An EMNLP Industry Track paper should not be written only as a traditional research paper focused on model performance.

The main story should be:

> We solved a real NLP/LLM problem in a practical setting, and the solution teaches something useful to the NLP community.

The paper should be **problem-first**, not model-first.

Instead of starting with:

> We fine-tuned model X using method Y and achieved Z improvement.

Write more like:

> In a production setting, users needed reliable detection across noisy, long-form documents where standard model outputs were difficult to trust. To address this, we developed a pipeline combining X, Y, and Z. This design reduced the practical problem while maintaining deployment constraints, and our results provide lessons for similar real-world NLP systems.

---

## What Makes an Industry Track Paper Strong?

A strong Industry Track paper usually includes:

| Writing Aspect | What It Should Do |
|---|---|
| Real-world motivation | Start with a concrete industry problem, not just a model gap. |
| Clear use case | Explain who uses the system, what workflow it supports, and why NLP/LLMs are needed. |
| Practical constraints | Discuss latency, cost, scale, privacy, noisy data, user trust, hallucination, deployment limits, or business constraints. |
| Technical contribution | Show what you built or changed technically, but connect every technical choice to the real-world problem. |
| Evaluation beyond benchmarks | Include business/product metrics, human review, user feedback, deployment performance, or failure analysis. |
| Lessons learned | Share practical insights that other researchers or industry teams can reuse. |
| Honest limitations | Be clear about what did not work, where the system fails, and what is not generalizable. |

---

## The Ideal Paper Story

A strong Industry Track paper often follows this logic:

> Many existing NLP/LLM methods work well in controlled benchmarks, but they fail or become difficult to use in real-world settings because of [constraint]. In this paper, we describe [system/method] developed for [real user/workflow]. We show how we handled [data, scale, cost, latency, privacy, evaluation, reliability]. Our results show [impact], and we share lessons from deployment that may help others building similar systems.

---

## Questions Your Paper Should Answer

Before writing, make sure your paper can answer these questions clearly.

### 1. What real problem are you solving?

Example:

> Users need to evaluate long documents for originality, AI involvement, or citation quality at scale.

Avoid vague framing like:

> LLM detection is important.

Make the problem concrete:

> Educators and content teams need reliable document-level signals across many uploaded files, but current workflows are slow, hard to interpret, and difficult to trust at scale.

---

### 2. Why is this problem hard in practice?

Do not only explain why it is technically hard. Explain why it is hard in the real world.

Possible real-world challenges:

- Long documents
- Noisy user-uploaded files
- Multiple file formats
- Ambiguous or mixed human/AI content
- Need for fast response time
- Privacy expectations
- Limited labeled data
- High false-positive risk
- User trust and explainability
- Cost of running LLM-based evaluation
- Need to scale to many users or documents

Example:

> The challenge is not only detecting AI-generated text, but doing so across long, noisy, user-uploaded documents while preserving speed, interpretability, and user trust.

---

### 3. What did you build?

Your contribution can be a:

- Deployed NLP/LLM system
- Production pipeline
- Human-in-the-loop workflow
- Evaluation framework
- Dataset creation or annotation process
- Model adaptation method
- Monitoring or quality-control system
- Retrieval or ranking system
- Prompting/evaluation workflow
- Hybrid rule-based and ML system
- Case study of a real deployment

You do **not** always need a new state-of-the-art model. For Industry Track, a practical system or evaluation approach can be valuable if it teaches something useful.

---

### 4. What is new or useful?

Your novelty may come from:

- Applying NLP/LLMs to a difficult real-world workflow
- Combining models, rules, and human review in a practical way
- Designing evaluation for a messy real-world task
- Showing deployment tradeoffs
- Improving cost, speed, reliability, or interpretability
- Sharing lessons from failures
- Showing how users actually interact with the system
- Creating a reproducible framework for a practical problem

A good novelty statement could be:

> The contribution of this paper is not a new foundation model, but a practical framework for deploying and evaluating LLM-assisted document analysis in a high-trust user workflow.

---

### 5. How do you prove it helped?

Use both technical and practical evidence.

Possible technical metrics:

- Accuracy
- Precision
- Recall
- F1
- False-positive rate
- Human agreement
- Calibration
- Robustness across document types
- Error rate by category

Possible industry/product metrics:

- Latency reduction
- Cost reduction
- User adoption
- Workflow completion rate
- Time saved
- Reduction in manual review
- User trust score
- Support ticket reduction
- Retention or repeat usage
- A/B test results
- Qualitative user feedback

Example:

> We evaluate the system using both classification performance and deployment-focused metrics, including latency, reviewer agreement, and user workflow completion.

---

### 6. What can others learn from it?

This is especially important for Industry Track.

Your paper should include lessons such as:

- What worked better than expected
- What failed
- What tradeoffs mattered most
- What evaluation methods were misleading
- What users misunderstood
- What monitoring was necessary
- What design decisions improved trust
- What constraints affected model choice

Example:

> Our deployment showed that improving raw model accuracy was less important than reducing uncertain outputs and making results interpretable to users.

---

## Recommended Section Flow

You can use this structure even if your official template is already set.

| Section | What to Write |
|---|---|
| Abstract | Summarize the real-world problem, system, evaluation, impact, and lessons. |
| Introduction | Problem → why it matters → why existing solutions are insufficient → your contribution. |
| Industry Context / Use Case | Describe the users, workflow, real-world constraints, and deployment environment. |
| System or Method | Explain the model, pipeline, architecture, or workflow clearly. |
| Evaluation | Combine technical metrics with practical/product/deployment metrics. |
| Deployment or Case Study | Show how the system works in practice, including examples or real scenarios. |
| Lessons Learned | Discuss tradeoffs, failures, user behavior, and reusable insights. |
| Related Work | Compare with academic and industry approaches. |
| Limitations | Be honest about data, evaluation, generalizability, risks, and deployment limits. |
| Ethical Considerations | Discuss privacy, fairness, bias, misuse, sensitive data, and user impact when relevant. |

---

## How to Write the Introduction

A good Industry Track introduction should move through this order:

1. Real-world problem
2. Why the problem matters
3. Why existing approaches are not enough
4. Practical challenges
5. What you built
6. How you evaluated it
7. Key results and lessons
8. Contributions

### Example Introduction Structure

> Organizations increasingly rely on NLP and LLM-based systems to support [task]. However, deploying these systems in real-world workflows remains challenging because [constraint 1], [constraint 2], and [constraint 3].
>
> In our setting, [user group] needs to [real workflow]. Existing approaches are insufficient because [reason].
>
> We present [system/method], a [brief description] designed for [real-world setting]. The system addresses [challenge] through [main design choices].
>
> We evaluate the system using [technical metrics] and [deployment/product metrics]. Our results show [impact].
>
> This paper contributes: (1) [system/framework], (2) [evaluation method], and (3) [deployment lessons].

---

## How to Frame Contributions

A strong contribution list for Industry Track might look like this:

> Our contributions are:
>
> 1. We present a deployed NLP/LLM system for [real-world task].
> 2. We describe practical design decisions for handling [constraint 1], [constraint 2], and [constraint 3].
> 3. We evaluate the system using both [technical metric] and [real-world/user/product metric].
> 4. We share deployment lessons and limitations that can guide similar industry applications.

Avoid contributions that are too generic:

> We improve performance.
>
> We use LLMs.
>
> We create a tool.

Make them specific:

> We introduce a deployment-oriented evaluation framework for document-level AI involvement detection, combining model confidence, human review, and user-facing interpretability signals.

---

## What to Emphasize in the Writing

### Emphasize practical constraints

Good:

> We selected a smaller model because the production workflow required results within 10 seconds per document batch.

Weak:

> We used a smaller model because it was faster.

### Emphasize user workflow

Good:

> The system was designed for educators reviewing many submissions at once, where interpretability and false-positive reduction were more important than maximizing recall alone.

Weak:

> The system classifies documents.

### Emphasize tradeoffs

Good:

> Increasing sensitivity improved recall but created more false positives, which reduced user trust in pilot testing.

Weak:

> The model had some errors.

### Emphasize deployment lessons

Good:

> We found that users relied more on highlighted evidence and uncertainty indicators than on the overall score alone.

Weak:

> We added explanations.

---

## Possible Paper Idea Patterns

Use one of these patterns to frame your idea.

### Pattern 1: Deployed System Paper

> We built and deployed an NLP/LLM system for [task] in [industry setting]. The paper explains the architecture, deployment constraints, evaluation, and lessons learned.

Best for:

- Real product features
- Production systems
- User-facing NLP tools

---

### Pattern 2: Evaluation Framework Paper

> We created a practical evaluation framework for [task], because standard benchmarks do not capture real-world constraints such as [constraints].

Best for:

- AI detection evaluation
- Long-document assessment
- Trust and interpretability
- Human-machine agreement

---

### Pattern 3: Human-in-the-Loop Workflow Paper

> We designed a workflow where NLP/LLM outputs support human decision-making rather than replacing it.

Best for:

- High-trust domains
- Education
- Compliance-like review
- Editorial or research workflows

---

### Pattern 4: Cost/Latency/Scale Optimization Paper

> We show how to make an NLP/LLM system practical at scale by reducing cost, latency, or compute while preserving quality.

Best for:

- LLM deployment
- Batch processing
- Production optimization
- Model routing

---

### Pattern 5: Failure Analysis / Lessons Learned Paper

> We analyze why a real-world NLP/LLM system failed or struggled, and what design changes improved it.

Best for:

- Honest industry case studies
- Evaluation gaps
- User trust problems
- Model reliability issues

---

## Strong Sentence Templates

### Problem sentence

> In real-world [domain] workflows, [user group] must [task], but existing NLP systems struggle because [practical constraint].

### Gap sentence

> Although prior work has focused on [academic focus], less attention has been paid to [industry/practical challenge].

### System sentence

> We present [system name], a production-oriented NLP/LLM pipeline designed to [task] under [constraints].

### Evaluation sentence

> We evaluate the system using both offline model metrics and deployment-focused measures, including [metrics].

### Lesson sentence

> Our deployment suggests that [practical lesson], which has implications for [broader NLP/LLM application].

### Contribution sentence

> This paper contributes a practical framework for [task], an evaluation methodology for [challenge], and lessons from deploying the system in [setting].

---

## Common Mistakes to Avoid

| Mistake | Why It Hurts |
|---|---|
| Writing only about model performance | Industry Track expects practical relevance and deployment insight. |
| Hiding the real use case | Reviewers need to understand why the work matters. |
| Ignoring constraints | Real-world constraints are part of the contribution. |
| Using only benchmark metrics | Industry papers need practical evidence too. |
| No lessons learned | The paper may feel like a product report rather than a research contribution. |
| Overclaiming generalizability | Be honest about where the system applies. |
| Too much business language | Keep it scientific and technical, not like a marketing document. |
| Too much engineering detail | Include engineering details only when they explain the NLP contribution or deployment challenge. |

---

## Final Test for Your Paper Idea

Your paper idea is probably strong if you can complete this sentence:

> This paper is useful because it teaches the NLP community how to build, evaluate, or deploy [system/method] in a real-world setting where [constraint] makes the problem difficult.

Example:

> This paper is useful because it teaches the NLP community how to evaluate document-level AI detection systems in real-world educational workflows where long documents, mixed authorship, and user trust make the problem difficult.

---

## One-Sentence Summary

For EMNLP Industry Track, write your paper as a **real-world NLP/LLM case study with technical depth, practical constraints, strong evaluation, and reusable lessons**, not just as a model-improvement paper.
