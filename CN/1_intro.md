## Who is this document for?

This document is for engineers and researchers (both individuals and teams)
interested in **maximizing the performance of deep learning models**. We assume
basic knowledge of machine learning and deep learning concepts.

Our emphasis is on the **process of hyperparameter tuning**. We touch on other
aspects of deep learning training, such as pipeline implementation and
optimization, but our treatment of those aspects is not intended to be complete.

We assume the machine learning problem is a supervised learning problem or
something that looks a lot like one (e.g. self-supervised). That said, some of
the prescriptions in this document may also apply to other types of problems.

## Why a tuning playbook?

Currently, there is an astonishing amount of toil and guesswork involved in
actually getting deep neural networks to work well in practice. Even worse, the
actual recipes people use to get good results with deep learning are rarely
documented. Papers gloss over the process that led to their final results in
order to present a cleaner story, and machine learning engineers working on
commercial problems rarely have time to take a step back and generalize their
process. Textbooks tend to eschew practical guidance and prioritize fundamental
principles, even if their authors have the necessary experience in applied work
to provide useful advice. When preparing to create this document, we couldn't
find any comprehensive attempt to actually explain *how to get good results with
deep learning*. Instead, we found snippets of advice in blog posts and on social
media, tricks peeking out of the appendix of research papers, occasional case
studies about one particular project or pipeline, and a lot of confusion. There
is a vast gulf between the results achieved by deep learning experts and less
skilled practitioners using superficially similar methods. At the same time,
these very experts readily admit some of what they do might not be
well-justified. As deep learning matures and has a larger impact on the world,
the community needs more resources covering useful recipes, including all the
practical details that can be so critical for obtaining good results.

We are a team of five researchers and engineers who have worked in deep learning
for many years, some of us since as early as 2006. We have applied deep learning
to problems in everything from speech recognition to astronomy, and learned a
lot along the way. This document grew out of our own experience training neural
networks, teaching new machine learning engineers, and advising our colleagues
on the practice of deep learning. Although it has been gratifying to see deep
learning go from a machine learning approach practiced by a handful of academic
labs to a technology powering products used by billions of people, deep learning
is still in its infancy as an engineering discipline and we hope this document
encourages others to help systematize the field's experimental protocols.

This document came about as we tried to crystalize our own approach to deep
learning and thus it represents the opinions of the authors at the time of
writing, not any sort of objective truth. Our own struggles with hyperparameter
tuning made it a particular focus of our guidance, but we also cover other
important issues we have encountered in our work (or seen go wrong). Our
intention is for this work to be a living document that grows and evolves as our
beliefs change. For example, the material on debugging and mitigating training
failures would not have been possible for us to write two years ago since it is
based on recent results and ongoing investigations. Inevitably, some of our
advice will need to be updated to account for new results and improved
workflows. We do not know the *optimal* deep learning recipe, but until the
community starts writing down and debating different procedures, we cannot hope
to find it. To that end, we would encourage readers who find issues with our
advice to produce alternative recommendations, along with convincing evidence,
so we can update the playbook. We would also love to see alternative guides and
playbooks that might have different recommendations so we can work towards best
practices as a community. Finally, any sections marked with a 🤖 emoji are places
we would like to do more research. Only after trying to write this playbook did
it become completely clear how many interesting and neglected research questions
can be found in the deep learning practitioner's workflow.

## Guide for starting a new project

Many of the decisions we make over the course of tuning can be made once at the
beginning of a project and only occasionally revisited when circumstances
change.

Our guidance below makes the following assumptions:

-   Enough of the essential work of problem formulation, data cleaning, etc. has
    already been done that spending time on the model architecture and training
    configuration makes sense.
-   There is already a pipeline set up that does training and evaluation, and it
    is easy to execute training and prediction jobs for various models of
    interest.
-   The appropriate metrics have been selected and implemented. These should be
    as representative as possible of what would be measured in the deployed
    environment.

### Choosing the model architecture

***Summary:*** *When starting a new project, try to reuse a model that already
works.*

-   Choose a well established, commonly used model architecture to get working
    first. It is always possible to build a custom model later.
-   Model architectures typically have various hyperparameters that determine
    the model's size and other details (e.g. number of layers, layer width, type
    of activation function).
    -   Thus, choosing the architecture really means choosing a family of
        different models (one for each setting of the model hyperparameters).
    -   We will consider the problem of choosing the model hyperparameters in
        [Choosing the initial configuration](#choosing-the-initial-configuration)
        and
        [A scientific approach to improving model performance](#a-scientific-approach-to-improving-model-performance).
-   When possible, try to find a paper that tackles something as close as
    possible to the problem at hand and reproduce that model as a starting
    point.

### Choosing the optimizer

***Summary:*** *Start with the most popular optimizer for the type of problem at
hand.*

-   No optimizer is the "best" across all types of machine learning problems and
    model architectures. Even just
    [comparing the performance of optimizers is a difficult task](https://arxiv.org/abs/1910.05446).
    🤖
-   We recommend sticking with well-established, popular optimizers, especially
    when starting a new project.
    -   Ideally, choose the most popular optimizer used for the same type of
        problem.
-   Be prepared to give attention to **\*****all****\*** hyperparameters of the
    chosen optimizer.
    -   Optimizers with more hyperparameters may require more tuning effort to
        find the best configuration.
    -   This is particularly relevant in the beginning stages of a project when
        we are trying to find the best values of various other hyperparameters
        (e.g. architecture hyperparameters) while treating optimizer
        hyperparameters as
        [nuisance parameters](#identifying-scientific-nuisance-and-fixed-hyperparameters).
    -   It may be preferable to start with a simpler optimizer (e.g. SGD with
        fixed momentum or Adam with fixed $\epsilon$, $\beta_{1}$, and
        $\beta_{2}$) in the initial stages of the project and switch to a more
        general optimizer later.
-   Well-established optimizers that we like include (but are not limited to):
    -   [SGD with momentum](#what-are-the-update-rules-for-all-the-popular-optimization-algorithms)
        (we like the Nesterov variant)
    -   [Adam and NAdam](#what-are-the-update-rules-for-all-the-popular-optimization-algorithms),
        which are more general than SGD with momentum. Note that Adam has 4
        tunable hyperparameters
        [and they can all matter](https://arxiv.org/abs/1910.05446)!
        -   See
            [How should Adam's hyperparameters be tuned?](#how-should-adams-hyperparameters-be-tuned)

### Choosing the batch size

***Summary:*** *The batch size governs the training speed and shouldn't be used
to directly tune the validation set performance. Often, the ideal batch size
will be the largest batch size supported by the available hardware.*

-   The batch size is a key factor in determining the *training time* and
    *computing resource consumption*.
-   Increasing the batch size will often reduce the training time. This can be
    highly beneficial because it, e.g.:
    -   Allows hyperparameters to be tuned more thoroughly within a fixed time
        interval, potentially resulting in a better final model.
    -   Reduces the latency of the development cycle, allowing new ideas to be
        tested more frequently.
-   Increasing the batch size may either decrease, increase, or not change the
    resource consumption.
-   The batch size should *not be* treated as a tunable hyperparameter for
    validation set performance.
    -   As long as all hyperparameters are well-tuned (especially the learning
        rate and regularization hyperparameters) and the number of training
        steps is sufficient, the same final performance should be attainable
        using any batch size (see
        [Shallue et al. 2018](https://arxiv.org/abs/1811.03600)).
    -   Please see [Why shouldn't the batch size be tuned to directly improve
        validation set
        performance?](#why-shouldnt-the-batch-size-be-tuned-to-directly-improve-validation-set-performance)

#### Determining the feasible batch sizes and estimating training throughput


<details><summary><em>[Click to expand]</em></summary>

<br>

-   For a given model and optimizer, there will typically be a range of batch
    sizes supported by the available hardware. The limiting factor is usually
    accelerator memory.
-   Unfortunately, it can be difficult to calculate which batch sizes will fit
    in memory without running, or at least compiling, the full training program.
-   The easiest solution is usually to run training jobs at different batch
    sizes (e.g. increasing powers of 2) for a small number of steps until one of
    the jobs exceeds the available memory.
-   For each batch size, we should train for long enough to get a reliable
    estimate of the *training throughput*

<p align="center">training throughput = (# examples processed per second)</p>

<p align="center">or, equivalently, the <em>time per step</em>.</p>

<p align="center">time per step = (batch size) / (training throughput)</p>

-   When the accelerators aren't yet saturated, if the batch size doubles, the
    training throughput should also double (or at least nearly double).
    Equivalently, the time per step should be constant (or at least nearly
    constant) as the batch size increases.
-   If this is not the case then the training pipeline has a bottleneck such as
    I/O or synchronization between compute nodes. This may be worth diagnosing
    and correcting before proceeding.
-   If the training throughput increases only up to some maximum batch size,
    then we should only consider batch sizes up to that maximum batch size, even
    if a larger batch size is supported by the hardware.
    -   All benefits of using a larger batch size assume the training throughput
        increases. If it doesn't, fix the bottleneck or use the smaller batch
        size.
    -   **Gradient accumulation** simulates a larger batch size than the
        hardware can support and therefore does not provide any throughput
        benefits. It should generally be avoided in applied work.
-   These steps may need to be repeated every time the model or optimizer is
    changed (e.g. a different model architecture may allow a larger batch size
    to fit in memory).

</details>

#### Choosing the batch size to minimize training time

<details><summary><em>[Click to expand]</em></summary>

<br>


<p align="center">Training time = (time per step) x (total number of steps)</p>

-   We can often consider the time per step to be approximately constant for all
    feasible batch sizes. This is true when there is no overhead from parallel
    computations and all training bottlenecks have been diagnosed and corrected
    (see the
    [previous section](#determining-the-feasible-batch-sizes-and-estimating-training-throughput)
    for how to identify training bottlenecks). In practice, there is usually at
    least some overhead from increasing the batch size.
-   As the batch size increases, the total number of steps needed to reach a
    fixed performance goal typically decreases (provided all relevant
    hyperparameters are re-tuned when the batch size is changed;
    [Shallue et al. 2018](https://arxiv.org/abs/1811.03600)).
    -   E.g. Doubling the batch size might halve the total number of steps
        required. This is called **perfect scaling**.
    -   Perfect scaling holds for all batch sizes up to a critical batch size,
        beyond which one achieves diminishing returns.
    -   Eventually, increasing the batch size no longer reduces the number of
        training steps (but never increases it).
-   Therefore, the batch size that minimizes training time is usually the
    largest batch size that still provides a reduction in the number of training
    steps required.
    -   This batch size depends on the dataset, model, and optimizer, and it is
        an open problem how to calculate it other than finding it experimentally
        for every new problem. 🤖
    -   When comparing batch sizes, beware the distinction between an example
        budget/[epoch](https://developers.google.com/machine-learning/glossary#epoch)
        budget (running all experiments while fixing the number of training
        example presentations) and a step budget (running all experiments with
        the number of training steps fixed).
        -   Comparing batch sizes with an epoch budget only probes the perfect
            scaling regime, even when larger batch sizes might still provide a
            meaningful speedup by reducing the number of training steps
            required.
    -   Often, the largest batch size supported by the available hardware will
        be smaller than the critical batch size. Therefore, a good rule of thumb
        (without running any experiments) is to use the largest batch size
        possible.
-   There is no point in using a larger batch size if it ends up increasing the
    training time.

</details>

#### Choosing the batch size to minimize resource consumption

<details><summary><em>[Click to expand]</em></summary>

<br>


-   There are two types of resource costs associated with increasing the batch
    size:
    1.  *Upfront costs*, e.g. purchasing new hardware or rewriting the training
        pipeline to implement multi-GPU / multi-TPU training.
    2.  *Usage costs*, e.g. billing against the team's resource budgets, billing
        from a cloud provider, electricity / maintenance costs.
-   If there are significant upfront costs to increasing the batch size, it
    might be better to defer increasing the batch size until the project has
    matured and it is easier to assess the cost-benefit tradeoff. Implementing
    multi-host parallel training programs can introduce
    [bugs](#considerations-for-multi-host-pipelines) and
    [subtle issues](#batch-normalization-implementation-details) so it is
    probably better to start off with a simpler pipeline anyway. (On the other
    hand, a large speedup in training time might be very beneficial early in the
    process when a lot of tuning experiments are needed).
-   We refer to the total usage cost (which may include multiple different kinds
    of costs) as the "resource consumption". We can break down the resource
    consumption into the following components:

<p align="center">Resource consumption = (resource consumption per step) x (total number of steps)</p>

-   Increasing the batch size usually allows us to
    [reduce the total number of steps](#choosing-the-batch-size-to-minimize-training-time).
    Whether the resource consumption increases or decreases will depend on how
    the consumption per step changes.
    -   Increasing the batch size might *decrease* the resource consumption. For
        example, if each step with the larger batch size can be run on the same
        hardware as the smaller batch size (with only a small increase in time
        per step), then any increase in the resource consumption per step might
        be outweighed by the decrease in the number of steps.
    -   Increasing the batch size might *not change* the resource consumption.
        For example, if doubling the batch size halves the number of steps
        required and doubles the number of GPUs used, the total consumption (in
        terms of GPU-hours) will not change.
    -   Increasing the batch size might *increase* the resource consumption. For
        example, if increasing the batch size requires upgraded hardware, the
        increase in consumption per step might outweigh the reduction in the
        number of steps.

</details>

#### Changing the batch size requires re-tuning most hyperparameters

<details><summary><em>[Click to expand]</em></summary>

<br>


-   The optimal values of most hyperparameters are sensitive to the batch size.
    Therefore, changing the batch size typically requires starting the tuning
    process all over again.
-   The hyperparameters that interact most strongly with the batch size, and therefore are most important to tune separately for each batch size, are the optimizer hyperparameters (e.g. learning rate, momentum) and the regularization hyperparameters.
-   Keep this in mind when choosing the batch size at the start of a project. If
    you need to switch to a different batch size later on, it might be
    difficult, time consuming, and expensive to re-tune everything for the new
    batch size.

</details>

#### How batch norm interacts with the batch size

<details><summary><em>[Click to expand]</em></summary>

<br>


-   Batch norm is complicated and, in general, should use a different batch size
    than the gradient computation to compute statistics. See the
    [batch norm section](#batch-normalization-implementation-details) for a
    detailed discussion.

</details>

### Choosing the initial configuration

-   Before beginning hyperparameter tuning we must determine the starting point.
    This includes specifying (1) the model configuration (e.g. number of
    layers), (2) the optimizer hyperparameters (e.g. learning rate), and (3) the
    number of training steps.
-   Determining this initial configuration will require some manually configured
    training runs and trial-and-error.
-   Our guiding principle is to find a simple, relatively fast, relatively
    low-resource-consumption configuration that obtains a "reasonable" result.
    -   "Simple" means avoiding bells and whistles wherever possible; these can
        always be added later. Even if bells and whistles prove helpful down the
        road, adding them in the initial configuration risks wasting time tuning
        unhelpful features and/or baking in unnecessary complications.
        -   For example, start with a constant learning rate before adding fancy
            decay schedules.
    -   Choosing an initial configuration that is fast and consumes minimal
        resources will make hyperparameter tuning much more efficient.
        -   For example, start with a smaller model.
    -   "Reasonable" performance depends on the problem, but at minimum means
        that the trained model performs much better than random chance on the
        validation set (although it might be bad enough to not be worth
        deploying).
-   Choosing the number of training steps involves balancing the following
    tension:
    -   On the one hand, training for more steps can improve performance and
        makes hyperparameter tuning easier (see
        [Shallue et al. 2018](https://arxiv.org/abs/1811.03600)).
    -   On the other hand, training for fewer steps means that each training run
        is faster and uses fewer resources, boosting tuning efficiency by
        reducing the time between cycles and allowing more experiments to be run
        in parallel. Moreover, if an unnecessarily large step budget is chosen
        initially, it might be hard to change it down the road, e.g. once the
        learning rate schedule is tuned for that number of steps.