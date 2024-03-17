---
title: "Jack's Car Rental"
date: 2024-03-17
layout: single
---


## Background

One of my goals this year is to learn more about reinforcement learning. I've been reading through the book "Reinforcement Learning: An Introduction" by Sutton and Barto, and I've been working through the exercises. This post details a particular example from the book, and presents the results of my code to replicate the example. Along the way we describe the policy iteration algorithm.

From Sutton and Barto, 2018, Example 4.2: 

> Jack manages two locations for a nationwide car rental company. Each day, some number of customers arrive at each location to rent cars. If Jack has a car available, he rents it out and is credited $10 by the national company. If he is out of cars at that location, then the business is lost. Cars become available for renting the day after they are returned. To help ensure that cars are available where they are needed, Jack can move them between the two locations overnight, at a cost of \$2 per car moved. We assume that the number of cars requested and returned at each location are Poisson random variables, meaning that the probability that the number is n is $\frac{\lambda^n}{n!}e^{-\lambda}$, where $\lambda$ is the expected number. Suppose $\lambda$ is 3 and 4 for rental requests at the first and second locations and 3 and 2 for returns. To simplify the problem slightly, we assume that there can be no more than 20 cars at each location (any additional cars are returned to the nationwide company, and thus disappear from the problem) and a maximum of five cars can be moved from one location to the other in one night. We take the discount rate to be $\gamma = 0.9$ and consider the problem of finding the optimal policy for maximizing the expected total reward.

The state is the number of cars at each location. The action is the number of cars to move from one location to the other. The reward is the profit from renting cars minus the cost of moving cars.

We solve this using the policy iteration algorithm.


## Policy Iteration

Letting $\pi(a | s)$ denote the probability of taking action $a$ in state $s$ under policy $\pi$, the state-value function $v_{\pi}(s)$ is the expected return starting from state $s$, and then following policy $\pi$. Letting $p(s', r | s, a)$ denote the probability of transitioning to state $s'$ and receiving reward $r$ given that we are in state $s$ and take action $a$, and letting $\gamma$ denote the discount factor,
the Bellman Equation states:

$$v_{\pi}(s) = \sum_{s} \pi(a|s) \sum_{s', r} p(s', r|s, a)[r + \gamma v_{\pi}(s')] \tag{S\&B, 4.4}$$

which expresses the value of any state $s$ in terms of the expected value of the next state $s'$ and the expected reward $r$. 

Provided a policy $\pi$, we can convert this into an iterative algorithm to find the value function. We start with an initial guess for the value function (e.g., $v_{\pi}^0 = \mathbf{0}$), and then repeatedly apply the Bellman Equation to each state until the value function converges. Explicitly, starting from $n=0$ we compute

$$v_{\pi}^{n+1}(s) = \sum_{a} \pi(a|s) \sum_{s', r} p(s', r|s, a)[r + \gamma v_{\pi}^n(s')]$$

until $\text{max}_s |v_{\pi}^{n+1}(s) - v_{\pi}^n(s)| < \theta$ for some small $\theta$. This is called policy evaluation.

Once we have the value function, we can use it to improve the policy by acting greedily with respect to it. Explicitly, we can define a new policy $\pi'$ such that $\pi'(s) = \arg\max_a \sum_{s', r} p(s', r|s, a)[r + \gamma v_{\pi}(s')]$. This is called policy improvement. This new policy is guaranteed to be as good as or better than the old policy.

We can then iterate between policy evaluation and policy improvement until the policy converges. This is called policy iteration. The pseudocode is as follows:

1. Initialize $\pi$ arbitrarily
2. Repeat until policy converges:
    1. Policy Evaluation: 
        1. Initialize $v_{\pi}^0 = \mathbf{0}$ (or some other initial guess)
        2. Repeat until $|v_{\pi}^{n+1}(s) - v_{\pi}^n(s)| < \theta$ for all $s$:
            1. For each $s$:
                1. $v_{\pi}^{n+1}(s) = \sum_{a} \pi(a|s) \sum_{s', r} p(s', r|s, a)[r + \gamma v_{\pi}^n(s')]$
    2. Policy Improvement:
        1. For each $s$:
            1. $\pi'(s) = \arg\max_a \sum_{s', r} p(s', r|s, a)[r + \gamma v_{\pi}(s')]$
        2. If $\pi' = \pi$, then stop
        3. $\pi = \pi'$

### Results

The convergence of the first round of policy evaluation looks like the following:

![evaluation 0](/assets/images/jacks_car_rental/policy_evaluation.gif)

After one iteration of policy evaluation and policy improvement, we get the policy:

![policy 1](/assets/images/jacks_car_rental/pi_1.png)

After four iterations, we get the optimal policy:

![policy 4](/assets/images/jacks_car_rental/pi_4.png)


## Code

The code is available in my [GitHub repository](https://github.com/ericoden/ericoden.github.io/blob/main/self_study/reinforcement_learning/ch_04/ex_7/jacks_car_rental.ipynb).
