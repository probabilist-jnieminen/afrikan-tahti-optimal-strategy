# Reinforcement learning refresher, with Afrikan tähti in mind

This is a study note, not a design document. Part I reviews the core ideas of reinforcement learning (RL). Part II maps them onto this game. Part III goes into the more advanced techniques that could actually be useful in this project. Each section ends with a short **In this game** note where relevant.

Math is written in LaTeX ($\cdot$), which GitHub renders.

---

## Part I: Core concepts

### 1. The setting

An **agent** interacts with an **environment** in discrete steps $t = 0, 1, 2, \dots$:

1. The agent observes the state $s_t$.
2. The agent picks an action $a_t$.
3. The environment returns a reward $r_{t+1}$ and a next state $s_{t+1}$.

The agent's goal is to maximise the expected **return**, the (discounted) sum of future rewards:

$$G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}, \qquad 0 \le \gamma \le 1.$$

$\gamma < 1$ is needed for infinite-horizon tasks. **Episodic** tasks (games that always end) can use $\gamma = 1$.

A **policy** $\pi(a \mid s)$ maps states to (distributions over) actions. Policies can be:
- **deterministic** (always the same action in a state) or **stochastic** (a probability distribution over actions),
- **stationary** (independent of $t$) or not.

### 2. Markov decision processes (MDPs)

An MDP is a tuple $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$:
- $P(s' \mid s, a)$: the transition probabilities,
- $R(s, a, s')$: the reward.

The **Markov property**: the future depends on the past only through the current state. Choosing a state representation that is Markov is a modelling decision, and often the most important one.

**Value functions:**

$$V^\pi(s) = \mathbb{E}_\pi[G_t \mid s_t = s], \qquad Q^\pi(s, a) = \mathbb{E}_\pi[G_t \mid s_t = s, a_t = a].$$

**Bellman expectation equation** (recursive consistency):

$$V^\pi(s) = \sum_a \pi(a \mid s) \sum_{s'} P(s' \mid s, a)\,[R(s,a,s') + \gamma V^\pi(s')].$$

**Bellman optimality equation:**

$$V^*(s) = \max_a \sum_{s'} P(s' \mid s, a)\,[R(s,a,s') + \gamma V^*(s')], \qquad \pi^*(s) = \arg\max_a Q^*(s, a).$$

Two facts about finite MDPs:
- A deterministic, stationary optimal policy always exists. Randomising never helps in a single-agent MDP.
- The optimal value function is unique. Several optimal policies can share it.

**Stochastic shortest path (SSP).** A special case: costs are positive, there is an absorbing goal state, and $\gamma = 1$. "Minimise the expected number of turns until the Star is in Cairo or Tangier" is exactly an SSP.

**Partial observability (POMDP).** If the agent cannot see the full state, the right "state" is the **belief**, a distribution over hidden states. The belief state is Markov even when observations are not.

**In this game:** the disks are hidden from *everyone*, and no player has private information. So the belief over unopened disks is the same for all players, and it is fully determined by the public history: which circles are opened, and which disk types remain. Opening a disk is therefore equivalent to drawing uniformly from the remaining multiset. Formally:
- The game is a fully observable stochastic game, with no POMDP machinery needed.
- The Markov state must include the remaining disk counts, not just which circles are open.

### 3. Dynamic programming (planning with a known model)

When $P$ and $R$ are known and the state space is small enough to enumerate:

- **Policy evaluation:** iterate the Bellman expectation equation until $V^\pi$ converges.
- **Policy iteration:** evaluate $\pi$, act greedily with respect to $V^\pi$, repeat. It converges in few iterations.
- **Value iteration:** repeatedly apply the optimality operator,
  $$V_{k+1}(s) = \max_a \sum_{s'} P(s'\mid s,a)\,[R + \gamma V_k(s')].$$
  For $\gamma < 1$ it is a contraction, so it converges geometrically. For SSP it converges under mild "proper policy" conditions.
- Complexity: each sweep costs about $|\mathcal{S}| \cdot |\mathcal{A}| \cdot$ (branching factor). That is feasible up to about $10^7$–$10^8$ states on a laptop when the transitions are sparse.

**In this game:** we have the model, since we wrote the rules. The full game is far too large for dynamic programming: Etzell bounds it at about $10^{22}$ states for two players. But some subproblems are small enough to solve exactly (see §13). Exact solutions are gold: they are ground truth to test RL against.

### 4. Model-free prediction: Monte Carlo and temporal-difference (TD) learning

When we only *sample* from the environment:

- **Monte Carlo (MC):** $V(s) \leftarrow V(s) + \alpha\,[G_t - V(s)]$. The estimate is unbiased but has high variance, because it uses the full noisy return.
- **TD(0):** $V(s) \leftarrow V(s) + \alpha\,[r_{t+1} + \gamma V(s_{t+1}) - V(s)]$. This is **bootstrapping**: updating a guess from a guess. It has lower variance but is biased while $V$ is wrong. The bracket is the **TD error** $\delta_t$.
- **n-step returns and TD($\lambda$):** these interpolate between TD(0) and MC.
  - The $\lambda$-return is $G_t^\lambda = (1-\lambda)\sum_n \lambda^{n-1} G_t^{(n)}$.
  - **Eligibility traces** implement it online.
  - $\lambda = 0$ is TD(0) and $\lambda = 1$ is MC.

**The bias–variance trade-off** runs through all of RL. Very random games like this one push towards methods that reduce variance.

### 5. Model-free control

- **SARSA (on-policy TD control):** $Q(s,a) \leftarrow Q(s,a) + \alpha[r + \gamma Q(s', a') - Q(s,a)]$, where $a'$ is the action actually taken next.
- **Q-learning (off-policy):** the target is $r + \gamma \max_{a'} Q(s', a')$. It learns $Q^*$ while behaving exploratorily.
- **Exploration vs exploitation:**
  - $\varepsilon$-greedy, softmax/Boltzmann, optimism, UCB (in bandits and MCTS).
  - In dice games the environment explores a lot for you. TD-Gammon learned with purely greedy self-play.
- **On-policy vs off-policy:**
  - On-policy methods learn the value of the policy that generates the data.
  - Off-policy methods learn about a different (target) policy, using replay buffers, importance sampling, etc.

### 6. Function approximation

Tables do not scale, so we approximate $V_\theta$, $Q_\theta$ or $\pi_\theta$ with a parametrised function: linear in features, a tree ensemble, or a neural network.

- **Semi-gradient TD:** $\theta \leftarrow \theta + \alpha\,\delta_t \nabla_\theta V_\theta(s_t)$. It is "semi" because we do not differentiate through the target.
- **The deadly triad:** function approximation + bootstrapping + off-policy data can diverge. DQN's tricks (below) exist to tame it.
- **Feature design** still matters with neural networks. A small network with good inputs beats a large one with raw inputs, especially with limited compute.

### 7. Deep value-based RL (the DQN family)

- **DQN** (Mnih et al., 2015): Q-learning with a convolutional network, plus
  - **experience replay**, which breaks the correlation between consecutive samples and reuses data,
  - a **target network**, a slowly updated copy of the network that gives stable targets.
- **Double DQN:** the online network selects $\arg\max$ and the target network evaluates it. This reduces overestimation bias, which noise makes worse, and this game is noisy.
- **Dueling networks:** $Q(s,a) = V(s) + A(s,a) - \text{mean}_a A(s,a)$. This helps when many actions have similar values.
- **Prioritised replay, n-step targets, noisy nets, distributional heads:** combined, these make "Rainbow".
- **Action masking:** set $Q$ of illegal actions to $-\infty$ before taking the $\arg\max$.

### 8. Policy-gradient and actor-critic methods

We optimise $J(\theta) = \mathbb{E}_{\pi_\theta}[G_0]$ directly.

- **The policy gradient theorem:**
  $$\nabla_\theta J = \mathbb{E}\big[\nabla_\theta \log \pi_\theta(a_t \mid s_t)\, \Psi_t\big].$$
  - $\Psi_t = G_t$ gives REINFORCE.
  - $\Psi_t = G_t - b(s_t)$ subtracts a **baseline**. Any baseline that does not depend on the action keeps the estimate unbiased and lowers its variance.
  - $\Psi_t = A(s_t, a_t) = Q - V$ uses the **advantage**.
- **Actor-critic:** the actor $\pi_\theta$ is trained with the advantage estimated by a learned critic $V_\phi$.
- **GAE** (generalised advantage estimation, Schulman et al., 2016): an exponentially weighted sum of TD errors,
  $$\hat A_t = \sum_l (\gamma\lambda)^l \delta_{t+l}.$$
  The parameter $\lambda$ trades bias for variance, the same idea as TD($\lambda$).
- **PPO** (Schulman et al., 2017): the workhorse of modern RL.
  - It maximises the clipped surrogate objective
    $$L = \mathbb{E}\big[\min(\rho_t \hat A_t,\ \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon)\hat A_t)\big], \qquad \rho_t = \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\text{old}}(a_t\mid s_t)},$$
    plus a value loss and an entropy bonus.
  - It runs several epochs over each batch of fresh on-policy data.
  - It is robust, simple and handles discrete actions with masks well.
- **Invalid action masking in policy gradients:** set the logits of illegal actions to $-\infty$ before the softmax. Huang & Ontañón (2022) show that this is a valid policy gradient and works much better than penalising illegal actions.

**Value-based vs policy-based, in one line:** value methods are sample-efficient off-policy learners but can be unstable. Policy-gradient methods are stable and handle stochastic policies and large action spaces naturally, but they need more samples.

### 9. Model-based RL and planning

- **Planning at decision time:** use the model to look ahead from the current state, then act. Examples are expectimax and MCTS.
- **Background planning:** use the model to generate training data or improve a value function. Examples are Dyna and dynamic programming.
- **Learned models:** MuZero learns its own dynamics model. That is unnecessary here, because our simulator *is* a perfect model. Our advantage over typical RL settings is a perfect, cheap and fast simulator.

### 10. Multi-agent RL and game theory

- **Stochastic (Markov) games:** an MDP with several agents, each with its own reward.
  - Two-player zero-sum games (win = +1, loss = 0) have a well-defined **minimax value**, and optimal play is "solve for the worst case".
  - With 3 or more players there is no single "optimal" policy. The outcome depends on what the others do.
- **Nash equilibrium:** a profile of policies in which no player gains by deviating alone.
  - In two-player zero-sum games, all equilibria give the same value. Playing an equilibrium policy guarantees at least that value.
  - In n-player games, equilibria are not interchangeable, and "optimal" must be stated relative to assumptions about the opponents.
- **Best response:** the optimal policy against a *fixed* opponent policy. Against a fixed opponent, the problem is a single-agent MDP again, because the opponent becomes part of the environment.
- **Multi-player search assumptions:**
  - **paranoid:** everyone else is against me,
  - **max$^n$:** everyone maximises their own score.

  Etzell used paranoid search.
- **Self-play and population methods:**
  - **naive self-play:** train against the current self,
  - **fictitious play:** best-respond to the *average* of past opponents,
  - **PSRO** (Lanctot et al., 2017): a growing population; each new policy is a best response to a mixture of the existing ones.

  Naive self-play can cycle in games with rock-paper-scissors dynamics. Population methods avoid that.
- **Exploitability:** how much a best response gains against a policy. It measures distance from equilibrium, and in two-player zero-sum games it is the natural measure of "how good is this policy".

---

## Part II: Afrikan tähti as an RL problem

### 11. State, action, transition, reward

**State (Markov).** Everything needed to predict the future:
- **for each player:**
  - location (one of 220),
  - money, in units of £100,
  - holds the Star / a winning horseshoe,
  - held (Slave Coast countdown, Sahara, St Helena),
  - on a free voyage, on a paid voyage, or neither;
- **board:**
  - the set of opened circles,
  - the counts of remaining disk types,
  - whether Cape Town's bonus has been claimed,
  - whether the Star has been found;
- whose turn it is and the turn phase (before the roll, after the roll with value $r$).

**Turn structure as decision points.** A turn has *two* decision points with a chance event between them:

1. **Before the roll:** pay to open, roll to open, roll to move, or free sail.
2. **The die is rolled:** a chance node.
3. **After the roll:** choose a destination reachable in $\le r$ steps, a flight, or staying put.
4. **On arrival at a red circle:** pay to open or not. This decision can be folded into step 3 as separate actions ("go to X and open", "go to X and stop").

**Action space.** A clean, fixed-size encoding is "target location + modifiers": about 220 destinations × {stop, stop and open}, plus a few special actions. Each step comes with a **legal-action mask**. Fixed-size masked actions are what PPO and DQN want.

**Rewards:**
- **Single-player (Q1):** $-1$ per turn with $\gamma = 1$. The value is minus the expected number of turns: an SSP.
- **Multiplayer (Q2/Q3):** $+1$ for a win and $0$ otherwise, given at the end of the game. The value is the *win probability*. Sparse, but games are short (about 25 decisions per player).
- These two objectives produce **different policies**; see §17.

**Episode length.** In two-player games the thesis measured about 47 plies on average. Rare pathological "everyone waits" states (Etzell §3.4) mean the simulator needs a turn cap.

### 12. Why this game is a good RL testbed

- **A perfect, fast simulator:** no model learning is needed, and we can afford millions of games.
- **Very high outcome variance:** this is the main obstacle. The signal (a better policy) is small compared with the noise (dice and disks), so variance reduction matters more here than in most textbook problems.
- **Graph-structured movement and a fixed board:** an MLP over per-node features is enough. A graph neural network (GNN) would only pay off if we wanted generalisation across different boards.
- **The exact sub-solutions in §13:** we can measure how far RL is from optimal, which most RL projects cannot do.

---

## Part III: Techniques of special interest for this project

### 13. Exact solutions where the state space allows

**The return trip (single player, Star in hand).** The state is:
- location (220),
- money (about 0–70 levels),
- voyage/held flags,
- the Cape Town flag.

That is roughly $10^5$–$10^6$ states, so value iteration solves it exactly in seconds. The Bellman equation with the "roll first, then decide" structure is

$$V(s) = \min\Big\{\ \underbrace{1 + \tfrac16 \sum_{r=1}^{6} \min_{a \in \mathcal{A}(s, r)} V(s'_{a})}_{\text{roll and move or fly}},\ \ \underbrace{1 + V(s_{\text{free sail}})}_{\text{free sail}},\ \dots \Big\}, \qquad V(\text{goal}) = 0.$$

This gives the exact answer to "how do I get home fastest from anywhere, with any amount of money" as a map: expected turns home from each location. That makes a nice figure.

A modelling caveat: should the returning player still be allowed to open disks to raise money for flights? If yes, the opened set enters the state and the problem is no longer small. A clean option: solve exactly without opening, then measure with simulation whether opening ever helps.

**Reduced games.** Exact dynamic programming is possible for small variants: fewer disks, a subgraph of the board, or merged disk types. These give ground truth for checking that the RL pipeline finds the optimum where the optimum is known.

**Afterstates (post-decision states).** In dice games it is natural to value the state *after* the decision but *before* the chance event, as TD-Gammon did. Here: the state after choosing a move, before the next roll. Learning $V(\text{afterstate})$ instead of $Q(s, a)$:
- reduces the number of values to learn,
- separates the two kinds of chance (dice, disks) from the decision.

### 14. TD-Gammon: the closest historical analogue

Tesauro's TD-Gammon (1992–1995):
- a small neural network evaluating backgammon positions, which are dice-driven,
- trained by **TD($\lambda$) self-play** from scratch,
- choosing moves greedily by evaluating every legal afterstate after the roll.

It reached world-class level with no search beyond one ply. Backgammon's dice provided the exploration. Afrikan tähti is similar: a dice race with discrete choices after the roll. "TD-Gammon for Afrikan tähti" is a simple, well-founded baseline, and a good interview story, because it shows how RL handles stochastic games.

### 15. MCTS and search with a learned evaluation

- **MCTS (UCT):** builds a search tree with simulations, choosing actions by an upper confidence bound,
  $$\bar X_a + c\sqrt{\ln N / N_a}.$$
  Chance nodes (dice, disk reveals) are sampled. It works with a random or heuristic rollout policy and no learning at all.
- **Expert iteration / AlphaZero:** the search acts as a policy-improvement operator.
  - The network $(p_\theta, v_\theta)$ guides the search.
  - The search's visit counts become training targets for $p_\theta$.
  - The game outcomes become targets for $v_\theta$.
  - Stochastic MuZero (Antonoglou et al., 2022) extends this to chance events.
- **A sobering data point:** Etzell found that deeper expectiminimax search did *not* help (depth 1 ≈ depth 3). Possible readings:
  - the game is mostly luck,
  - the bottleneck was the hand-made evaluation,
  - shallow search cannot see the long-term effects.

  A learned value function plugged into the same search directly tests the second reading.

### 16. Reward shaping without changing the optimum

Sparse win/loss rewards make learning slow. **Potential-based shaping** (Ng, Harada & Russell, 1999) adds

$$F(s, s') = \gamma \Phi(s') - \Phi(s)$$

for any potential function $\Phi$. It provably leaves the optimal policy unchanged. Natural potentials here:
- minus the distance to the nearest unopened disk,
- the exact expected turns home from §13 (when holding the Star),
- money.

Any *other* shaping (e.g. "+0.1 per opened disk") can silently change what is optimal.

### 17. Win probability is not expected time: risk sensitivity

- Single-player Q1 minimises $\mathbb{E}[T]$, the expected number of turns.
- In a two-player race, what matters is $P(T_{\text{me}} < T_{\text{opp}})$. A player who is *behind* should prefer **high-variance** plays (long-shot flights, gambling on far-away disks). A player who is *ahead* should prefer safe ones.

So the single-player optimal policy is *not* the two-player optimal policy, even when the opponent does not interact. Quantifying this gap is a clean answer to Q3.

Tools:
- **Distributional RL** (C51, Bellemare et al., 2017; QR-DQN, Dabney et al., 2018): learns the full *distribution* of the return, not just its mean. For Q1 this gives the whole distribution of turns to finish. For Q2 it shows the variance–mean trade-off directly.
- **Exact computation for the endgame:** if both players hold a Star or horseshoe and are racing home, $P(\text{win})$ can be computed exactly from the two time-to-home distributions. It is another small, exactly solvable subproblem.

### 18. Multi-agent questions in this game

- **Interaction is weak but real:**
  - the race to the Star,
  - disks taken by the opponent are gone,
  - horseshoes become valuable only once the Star is found,
  - the Cape Town bonus,
  - the "who opens first" standoff (Etzell §3.4).
- **Q3 as a best-response computation:**
  - train a policy to be optimal against a *fixed* baseline opponent (an MDP again),
  - compare it with the single-player-optimal policy,
  - measure how much the best response gains over the single-player policy when facing that opponent.
- **Self-play vs fixed opponents:** with weak interaction, self-play should be stable. Still, evaluate against a *pool* of fixed baselines (Close, Fly, Frugal, Random, Etzell-style search) so the results can be read on their own.

### 19. Measuring skill vs luck (Q2)

This is where the project can say something new.

- **Pairwise win rates with confidence intervals:**
  - The standard error is about $\sqrt{p(1-p)/n}$.
  - Resolving a 1-percentage-point difference needs about $10^4$–$10^5$ games.
  - Use **Wilson** intervals rather than the normal approximation near 0 or 1.
- **Common random numbers / "duplicate" play:**
  - Replay the *same* disk layout and dice sequences with the policies swapped, as in duplicate bridge.
  - Luck is then shared and mostly cancels out, so the same precision needs far fewer games.
  - This requires the engine to have **separate random streams** for the disk layout and for each player's dice.
- **Control variates:**
  - Subtract a known-mean quantity that is correlated with the outcome, e.g. "how early the Star was found" or the expected disk value.
  - **AIVAT** (Burch et al., 2018) does this for poker agents using a value function. The same idea applies here.
- **Rating models:** Elo or Bradley–Terry fitted to all pairwise results places every policy on one scale.
- **A relative skill measure** (Borm & van der Genugten, 2001; Dreef, Borm & van der Genugten, 2004):
  - The **learning effect** is the result gap between a beginner and an optimal player.
  - The **random effect** is the gap between the optimal player and a fictitious player who *knows the chance outcomes in advance*.
  - $\text{skill} = \text{learning} / (\text{learning} + \text{random})$.
  - Our simulator can compute all three terms: "random" or "Close" as the beginner, the best policy we find as the optimum, and an oracle that sees the disks and dice.
  - The result is one number, "Afrikan tähti is X% skill". It is a natural headline for the report.

### 20. Making the learned policy understandable

A neural policy is opaque. To turn it into advice your sister could beat you with:

- **Policy distillation into a decision tree** (e.g. VIPER, Bastani et al., 2018): train a shallow tree to imitate the RL policy, then read off rules such as "if money ≥ £400 and the nearest disk is more than 6 steps away, fly".
- **Counterfactual tables:** in which situations does the learned policy disagree with "Close"? How much does each disagreement gain in win probability?
- **Maps:** the value of standing at each location, given money. The exact maps from §13 and the learned maps side by side.

### 21. Practical notes

- **Environment interfaces:**
  - **Gymnasium** for single-agent environments (`reset`, `step` → `obs, reward, terminated, truncated, info`).
  - **PettingZoo**'s AEC ("agent–environment cycle") API for turn-based multi-agent games.

  We don't have to depend on them, but matching their conventions makes the environment familiar to readers.
- **Vectorised environments:** step $N$ games at once and return arrays. With a Rust engine this keeps the Python↔Rust overhead negligible.
- **Seeding and reproducibility:**
  - Every experiment runs from a seed.
  - Report the mean ± a confidence interval over several training seeds, not the best run.
  - RL results vary a lot between seeds.
- **Debugging RL:**
  1. Test on the exactly solved subproblems first.
  2. Check that the value loss goes down and that the entropy does not collapse immediately.
  3. Plot win rate against fixed baselines during training.
  4. A policy that beats random but not "Close" usually has a bug or a reward or observation problem.
- **Starter implementations:**
  - CleanRL: single-file PPO and DQN, good for learning because everything is visible.
  - Stable-Baselines3 with sb3-contrib: `MaskablePPO`.

  The choice will be made in the design discussion.

---

## Where each technique fits in this project

| Question | Core approach | Ground truth / comparison | Advanced extension |
|---|---|---|---|
| Q1: fastest single-player finish | PPO or TD with afterstates, reward $-1$ per turn | Exact value iteration for the return trip; exact dynamic programming on reduced boards | Distributional RL for the full distribution of turns |
| Q2: how much does strategy matter | Baselines + RL policy, pairwise and duplicate evaluation | Etzell's numbers (Close vs Search, etc.) | Relative skill measure with an oracle player; AIVAT-style variance reduction |
| Q3: does a rival change the strategy | Best response to fixed opponents vs the single-player policy | Exact endgame race probabilities | Self-play / PSRO population; risk-sensitive behaviour when behind vs ahead |

## Quick self-test (interview-style)

1. Why does Q-learning overestimate, and how does Double DQN fix it?
2. Why does a baseline not bias the policy gradient?
3. What does the PPO clip protect against?
4. What is the deadly triad?
5. When is $\gamma = 1$ allowed?
6. What makes a state Markov? Is "my location and money" Markov in this game? (No: the remaining disk counts and the opponent's state matter.)
7. Why do on-policy methods need fresh data after each update?
8. What is the difference between a best response and a Nash equilibrium? Why is "optimal" ambiguous with 3 or more players?
9. Why might maximising win probability prefer a *higher* expected finishing time?
10. How do common random numbers reduce the number of games needed to compare two policies?

## Further reading

- **Sutton & Barto**, *Reinforcement Learning: An Introduction*, 2nd ed. (2018). Chapters 3–6 (MDPs, dynamic programming, MC, TD), 9–10 (function approximation), 13 (policy gradient) and 16.1 (TD-Gammon). Free online.
- **OpenAI Spinning Up** (2018): a compact introduction to deep RL, with PPO derivations.
- **Tesauro**, "Temporal Difference Learning and TD-Gammon", *Communications of the ACM* (1995).
- **Mnih et al.**, "Human-level control through deep reinforcement learning", *Nature* (2015). **van Hasselt et al.**, Double DQN (2016). **Wang et al.**, Dueling DQN (2016).
- **Schulman et al.**, GAE (2016) and PPO (2017).
- **Huang & Ontañón**, "A Closer Look at Invalid Action Masking in Policy Gradient Algorithms" (FLAIRS 2022).
- **Ng, Harada & Russell**, "Policy invariance under reward transformations" (ICML 1999).
- **Bellemare, Dabney & Munos**, "A Distributional Perspective on RL" (2017). **Dabney et al.**, QR-DQN (2018).
- **Browne et al.**, "A Survey of Monte Carlo Tree Search Methods" (2012). **Anthony, Tian & Barber**, "Thinking Fast and Slow with Deep Learning and Tree Search" (expert iteration, 2017). **Silver et al.**, AlphaZero (*Science*, 2018). **Antonoglou et al.**, Stochastic MuZero (ICLR 2022).
- **Lanctot et al.**, "A Unified Game-Theoretic Approach to Multiagent RL" (PSRO, NeurIPS 2017).
- **Sturtevant & Korf** (2000), paranoid search. **Luckhardt & Irani** (1986), max$^n$.
- **Burch et al.**, "AIVAT: A New Variance Reduction Technique for Agent Evaluation in Imperfect Information Games" (AAAI 2018).
- **Borm & van der Genugten**, "On a relative measure of skill for games with chance elements", *TOP* 9 (2001). **Dreef, Borm & van der Genugten**, "Measuring skill in games: several approaches discussed", *Math. Methods of OR* 59 (2004).
- **Bastani, Pu & Solar-Lezama**, "Verifiable Reinforcement Learning via Policy Extraction" (VIPER, NeurIPS 2018).
- **Etzell**, *On the gameplay of Afrikan tähti* (Aalto, 2025), in `docs/context/`.
