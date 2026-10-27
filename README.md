# Black-Box Optimisation Capstone: Finding the Maxima of Eight Unknown Functions

## NON-TECHNICAL EXPLANATION OF YOUR PROJECT

Imagine trying to find the best recipe, machine setting or drug dose when every test is slow and expensive, and you can't see how the process works inside. This project tackles eight such "black box" problems, ranging from locating a radiation source to tuning a machine-learning model. For each one I get a handful of past results and just one new test per week. Rather than guessing, I build a statistical "map" of each problem from the results so far. The map shows both where the best answer probably lies and where I am still uncertain. I then choose each week's test to balance improving on known good settings against exploring the unknown.

## DATA

The data was provided by the capstone programme as part of a Bayesian-optimisation-style challenge. It is synthetic: each function is an unknown mathematical function written by the course team to imitate a real-world problem. No external datasets were used.

| Function | Inputs | Starting points | Real-world analogy (from the brief)                                             |
| -------- | ------ | --------------- | ------------------------------------------------------------------------------- |
| 1        | 2      | 10              | Detecting contamination (e.g. radiation) sources: signal only near a source     |
| 2        | 2      | 10              | Noisy log-likelihood of a mystery ML model, with local optima                   |
| 3        | 3      | 15              | Drug discovery: minimise adverse reactions (outputs negated)                    |
| 4        | 4      | 30              | Warehouse product placement: ML approximation of an expensive calculation       |
| 5        | 4      | 20              | Chemical process yield (typically one peak)                                     |
| 6        | 5      | 20              | Cake recipe: negative score from flavour, consistency, calories, waste and cost |
| 7        | 6      | 30              | Tuning six ML hyperparameters                                                   |
| 8        | 8      | 40              | Tuning eight ML hyperparameters (e.g. learning rate, dropout, optimiser type)   |

- **Format:** for each function, `initial_inputs.npy` (an n × d array, with every input in [0, 1]) and `initial_outputs.npy` (n values). Every task is a **maximisation**.
- **Growth:** each week I submit one query per function through the portal, formatted as six-decimal values joined by hyphens (e.g. `0.419675-0.463269`), and receive one new output. All submissions and results are recorded in the `HISTORY` cell of the notebook.
- **Notable properties of the starting data:**
  - Function 1 is almost zero everywhere (1e-124 to 1e-15).
  - Function 5 spans four orders of magnitude (0.1 to 1089).
  - Function 8 is close to linear (a straight-line fit explains 90% of the variance).

![Initial data for the 2D functions](images/initial_data_2d.png)

_Data for Functions 1 and 2 after Week 4. Colour shows the output: log10|y| for F1, y for F2. The orange circle marks the best point._

## MODEL

**Primary model: a Gaussian Process (GP) surrogate with Bayesian optimisation.**

A GP predicts the output at any input **together with an uncertainty**. With only 10–40 expensive observations, that uncertainty is what makes principled exploration possible, so a GP was the natural choice. My configuration:

- **ARD kernel** (a separate length-scale for each input), so the learned length-scales show which inputs matter. Weeks 1–2 used Matern 5/2 everywhere. **From Week 3 the kernel is chosen per function** from Matern 3/2, Matern 5/2 and RBF, by log marginal likelihood. In Week 3 the data chose RBF for Functions 2, 3, 5, 6 and 7, and Matern 5/2 for Functions 4 and 8.
- **A minimum noise level (from Week 4).** The noise term may not fall below 0.03 (in normalised units). Without this floor the GP passed almost exactly through every point and was overconfident: three predictions missed by 4–5 standard deviations. A backtest on Weeks 2–3 cut the typical miss from 3.0 to 1.3 standard deviations, and in Week 4 the typical miss was 1.0. Function 4, whose outputs span -32 to +0.7, uses a lower floor (0.001) from Week 5, so that its peak is not blurred.
- **A noise-aware incumbent (from Week 5).** "Best point" means the observed point with the highest _model estimate_, not the single highest reading. Week 4 re-tests showed repeat readings at almost the same spot differ by as much as the gains being chased, so the highest reading is often a lucky draw (the winner's curse).
- **A white-noise term**, for noisy functions such as Function 2 and for numerical stability.
- **Normalised outputs**, plus transforms: **log(y)** for Function 5 because of its huge range. For Function 1, whose readings range from 1e-288 to 0.006, I used **log10|y|** in Week 2 and a sign-aware version from Week 4.
- **Leave-one-out validation** (Spearman correlation) to measure how far each function's GP can be trusted.

An **acquisition function** turns the GP's prediction and uncertainty into a score for each candidate point: Expected Improvement (EI) or Upper Confidence Bound (UCB). The best-scoring point is found in three steps:

1. **Score about 60,000 candidates**, drawn globally and around the best known points. Points that fall outside [0, 1] are reflected back inside.
2. **Refine the 10 best** with a gradient optimiser (multi-start L-BFGS-B).
3. **Anchor irrelevant inputs:** any input the model considers irrelevant is reset to its value at the best known point.

Steps 2 and 3, and the reflection in step 1, were added in Week 2 (see Results).

**Four alternative strategies are run as cross-checks every week**, because one model can be misled by so little data:

| Method                                 | Idea                                                                                                                                                | Why include it                                                       |
| -------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| TPE (Tree-structured Parzen Estimator) | Models the density of "good" and "bad" inputs                                                                                                       | Standard in hyperparameter-tuning tools, robust in higher dimensions |
| Random Forest (the idea behind SMAC)   | A tree ensemble, with the spread between trees as the uncertainty                                                                                   | Handles thresholds and discrete-like inputs                          |
| TuRBO-style trust region               | GP search in a box that grows after successes and shrinks after failures. From Week 3, inputs the model considers irrelevant stay at the best value | Built for high dimensions with small budgets                         |
| Linear trend-following                 | A step up the fitted gradient                                                                                                                       | A simple, transparent baseline                                       |

When the methods agree, I exploit confidently. When they disagree, I stay closer to what has worked.

![Proposals by method](images/method_comparison.png)

_Where each of the five methods proposes to query in Week 5. Overlapping lines mean the methods agree._

## HYPERPARAMETER OPTIMISATION

The project involves two layers of hyperparameters.

**1. Surrogate-model hyperparameters, fitted automatically.** The GP's amplitude, one length-scale per input and its noise level are all fitted by **maximising the marginal likelihood**, using L-BFGS with 15 random restarts to avoid poor local fits. From Week 3 the kernel family is also a fitted choice: the one with the highest marginal likelihood wins. From Week 4 the noise level has a lower bound of 0.03, chosen by backtesting calibration on Weeks 2–3. Length-scales are bounded to [0.02, 10] and noise to [1e-6, 0.1]. The fitted length-scales are useful in themselves: a short one marks an important input, and one at the upper bound marks an input the model considers irrelevant.

![Learned length-scales](images/lengthscales.png)

_Learned ARD length-scales after Week 4 (log scale). A shorter bar means a more important input. For Function 2 the model treats x2 as irrelevant, and for Function 3 it treats x1 and x2 as irrelevant._

**2. Optimisation-strategy hyperparameters, set per function and reviewed weekly.** These control the balance between exploration and exploitation. They are set by reasoning from each function's brief and data, not by automated tuning, because every evaluation costs a week.

| Setting          | Role                                             | Week 1          | Week 2              | Week 3            | Week 4             | Week 5                         |
| ---------------- | ------------------------------------------------ | --------------- | ------------------- | ----------------- | ------------------ | ------------------------------ |
| Kernel           | Smoothness assumption                            | Matern 5/2      | Matern 5/2          | Per function      | Per function       | Per function                   |
| Noise floor      | Minimum noise level                              | 1e-6            | 1e-6                | 1e-6              | 0.03               | 0.03 (F4: 0.001)               |
| Incumbent        | What counts as "best"                            | Highest reading | Highest reading     | Highest reading   | Highest reading    | Highest model estimate         |
| κ (UCB)          | Exploration strength                             | 2               | 1–1.5               | 1–2               | 0.5–1              | 3 for F2, 1.5 for F8, 1 for F7 |
| ξ (EI)           | Exploration strength                             | 0.01            | 0.005 for F4, F6    | 0.001 for F4      | 0.001–0.005        | 0.05 for F4, 0.01 otherwise    |
| Trust region     | Limits the search to ±`tr` around the best point | F5 only         | F3, F4, F7          | All except F1, F2 | ±0.05–0.2          | ±0.1–0.25; none for F2         |
| Output transform | Makes wide-ranging outputs easier to model       | log(y) for F5   | + log10\|y\| for F1 | Same              | F1: sign-aware log | Same                           |
| Exploration mode | Maximin space-filling                            | F1              | Off                 | F1 (whole space)  | F1 (upper right)   | F1 (around its best reading)   |

**Weekly adjustment rule:**

- Where a query improves on the best, I lower κ or ξ and shrink the search region (exploit).
- Where a function stalls for 2–3 weeks, I raise them (explore).
- To choose each final query, I follow the GP where it predicts well (LOO ≥ 0.7), take the consensus of the five methods where they agree, and use TuRBO where they disagree. I can override this with a written reason.

Functions 7 and 8 are themselves ML hyperparameter-tuning problems, so this project is also a small case study in **Bayesian hyperparameter optimisation**.

## RESULTS

_Status: Week 4 results received, Week 5 queries chosen. This section is updated each week._

### Best value per function so far

| Function | Starting best | Week 1  | Week 2  | Week 3  | Week 4   | Model's estimate | Change so far                       |
| -------- | ------------- | ------- | ------- | ------- | -------- | ---------------- | ----------------------------------- |
| 1        | 7.7e-16       | 7.7e-16 | 1.2e-13 | 1.2e-13 | 1.2e-13  | –                | Still effectively zero              |
| 2        | 0.611         | 0.611   | 0.611   | 0.611   | 0.611    | 0.570            | No gain (best reading partly noise) |
| 3        | -0.035        | -0.035  | -0.035  | -0.021  | -0.021   | -0.034           | Within noise of the start           |
| 4        | -4.03         | -0.379  | 0.712   | 0.712   | 0.712    | 0.493            | From -4.03 to about +0.5–0.7        |
| 5        | 1089          | 1639    | 2605    | 4269    | **4585** | 4311             | 4.2 times the start                 |
| 6        | -0.714        | -0.462  | -0.462  | -0.462  | -0.462   | -0.524           | About 30% closer to zero            |
| 7        | 1.365         | 1.365   | 1.462   | 1.687   | 1.687    | 1.566            | +15–24%                             |
| 8        | 9.60          | 9.891   | 9.891   | 9.891   | 9.891    | 9.887            | +3%                                 |

The **model's estimate** averages nearby readings, so it is a fairer measure of where each function really stands than the single best reading. The gap between the two is small for Functions 5, 7 and 8, and large for Functions 3 and 4, where noise inflated the best reading.

![Best value so far](images/progress.png)

_Best value observed so far for each function. Week 0 is the best of the starting data._

### Results against predictions

| Function | W1 result  | z    | W2 result | z        | W3 result  | z        | W4 result | z    |
| -------- | ---------- | ---- | --------- | -------- | ---------- | -------- | --------- | ---- |
| 1        | -0.0061    | –    | 1.2e-13   | -0.8     | 1.5e-288   | -5.9     | 4.9e-144  | –    |
| 2        | -0.026     | -1.8 | 0.593     | -0.3     | 0.450      | -1.1     | 0.506     | -1.9 |
| 3        | -0.138     | -1.0 | -0.048    | 0.0      | **-0.021** | +1.6     | -0.035    | -0.1 |
| 4        | **-0.379** | +2.0 | **0.712** | **+5.0** | 0.494      | +0.6     | 0.588     | +0.6 |
| 5        | **1639**   | -0.3 | **2605**  | +0.3     | **4269**   | +0.6     | **4585**  | +0.3 |
| 6        | **-0.462** | -0.8 | -0.517    | -0.7     | -0.871     | **-4.7** | -0.613    | -1.7 |
| 7        | 1.219      | -0.4 | **1.462** | +1.5     | **1.687**  | **+4.1** | 1.540     | -0.3 |
| 8        | **9.891**  | -1.3 | 9.859     | -1.0     | 9.849      | -1.2     | 9.826     | -0.5 |

Bold marks a new best reading. **12 of 32 queries improved the best: 4, 4, 3 and 1 per week.** After the Week 4 noise floor, every result except Function 1's fell within two standard deviations of its prediction (typical miss 1.0, against 3.0 before).

### What I have learned

- **Exploiting a clear trend with a reliable model works.** Function 5 has improved every week and is now 4.2 times its starting value. Function 4 went from -4.03 to about +0.5–0.7.
- **Noise sets the limit of local refinement.** Week 4 re-tests showed repeat readings at almost the same point differ by about ±0.05 (Function 2) and ±0.007 (Function 3), as much as the gains I was chasing. Several "best so far" values were lucky draws. From Week 5 the search is anchored on the model's estimate instead.
- **Small steps beat big ones, until the noise floor.** Steps of 0.02–0.26 gave most of the Week 2–3 gains. By Week 4, steps that small could only find noise.
- **Agreement between methods is not proof.** All five methods agreed on Function 6's Week 3 query, and it was my worst miss.
- **A model can be calibrated on average and still overconfident.** Average z-scores looked fine in Weeks 2–3, but three misses of 4–5 standard deviations showed the uncertainty was too narrow near sharp peaks. A minimum noise level fixed this.
- **Failed explorations still rule things out.** Function 1 has nothing in its centre, top-left or top-right corners. Function 2 has no second peak at the high-x1 edge.

### Pipeline changes so far

| After  | Change                                                             | Why                                                                                            |
| ------ | ------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------- |
| Week 1 | Reflect candidates instead of clipping them                        | Clipping biased queries towards exactly 0 or 1                                                 |
| Week 1 | Multi-start L-BFGS-B refinement of the acquisition                 | Random candidates alone were imprecise in 6–8 dimensions                                       |
| Week 1 | Anchor irrelevant inputs to the best known point                   | They were otherwise set arbitrarily                                                            |
| Week 1 | Store and review predictions                                       | To track how well calibrated the model is                                                      |
| Week 2 | Choose the kernel per function by marginal likelihood              | Function 4's z = +5. For Function 5, the reliability score rose from 0.57 to 0.81              |
| Week 2 | TuRBO holds irrelevant inputs at the best value                    | It proposed far corners (the Function 3 override in Week 2)                                    |
| Week 3 | Minimum GP noise level of 0.03                                     | Three misses of 4–5 standard deviations. Typical miss: 3.0 → 1.3 (backtest), then 1.0 (Week 4) |
| Week 3 | Sign-aware transform for Function 1                                | log10\|y\| counted negative readings as good                                                   |
| Week 4 | Noise-aware incumbent: highest model estimate, not highest reading | Re-tests showed best readings inflated by noise                                                |
| Week 4 | Function 4 noise floor lowered to 0.001                            | Its wide output range made the shared floor blur its peak                                      |

### Queries submitted

| Function | Week 3 (method)                                                                | Week 4 (method)                                                                | Week 5 (method)                                                                |
| -------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------ |
| 1        | `0.051118-0.947227` (maximin)                                                  | `0.949988-0.949929` (maximin)                                                  | `0.450249-0.649384` (maximin, around best reading)                             |
| 2        | `0.727137-0.926564` (TuRBO)                                                    | `0.689710-0.926564` (GP, re-test)                                              | `0.000000-0.926564` (GP, override: explore)                                    |
| 3        | `0.492581-0.611593-0.362899` (TuRBO)                                           | `0.492581-0.611593-0.369207` (consensus, re-test)                              | `0.492581-0.811593-0.515252` (GP, override: explore)                           |
| 4        | `0.375309-0.431898-0.364200-0.431114` (GP)                                     | `0.407459-0.425565-0.346036-0.420348` (GP)                                     | `0.363444-0.443846-0.387839-0.431114` (GP)                                     |
| 5        | `0.239972-0.999999-0.986193-0.999999` (GP)                                     | `0.439972-0.999999-0.999999-0.999999` (GP)                                     | `0.639972-0.999999-0.999999-0.999999` (GP)                                     |
| 6        | `0.423548-0.096277-0.881854-0.999999-0.000000` (consensus)                     | `0.460127-0.029091-0.638806-0.797785-0.000000` (GP)                            | `0.472525-0.305141-0.478220-0.903175-0.000000` (GP)                            |
| 7        | `0.033992-0.491672-0.247422-0.279911-0.355814-0.698452` (TuRBO)                | `0.000000-0.491672-0.247422-0.315039-0.355814-0.739334` (TuRBO)                | `0.003010-0.491672-0.247422-0.291440-0.355814-0.698452` (TuRBO)                |
| 8        | `0.180785-0.000000-0.124313-0.000000-0.999999-0.234896-0.223487-0.462463` (GP) | `0.000000-0.041964-0.196966-0.000000-0.999999-0.474929-0.000000-0.362463` (GP) | `0.000000-0.000000-0.000000-0.000000-0.999999-0.724929-0.000000-0.612463` (GP) |

Week 1–2 queries are in the notebook history.

![Method proposals on 2D surfaces](images/method_comparison_2d.png)

_Week 5 proposals from all five methods on the 2D functions._

### Repository contents

| File                          | Description                                                                                                                                   |
| ----------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `capstone_bbo.ipynb`          | Full weekly pipeline: data, exploratory analysis, GP, acquisition, five-method comparison, final selection, progress tracking and reflections |
| `function_1/` … `function_8/` | Initial `.npy` data                                                                                                                           |
| `week_<N>_queries.csv`        | Queries chosen each week, with the method used and the model's prediction                                                                     |
| `images/`                     | Figures used in this README                                                                                                                   |

**To run it:** `pip install numpy pandas scipy scikit-learn matplotlib jupyter`, then open the notebook next to the `function_*` folders and run all cells.
