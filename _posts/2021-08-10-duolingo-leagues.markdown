---
title: "Duolingo Leagues"
date: 2021-08-10 09:06:32 -0400
layout: single
---

[Duolingo](https://www.duolingo.com/) pools users into ten leagues (from Bronze, Silver, ..., Diamond). Each week, the top XP-scoring members of each league are promoted and the bottom scoring-XP members are demoted (the specifics are [here](https://duolingo.fandom.com/wiki/League)). Assuming no users enter, this can be modeled as a discrete time Markov chain:

$$\mathbf{x}_{n+1} = \mathbf{A}\mathbf{x}_n$$

where $\mathbf{x}_i$ is the vector of population levels for each league in week
$i$ and $\mathbf{A}$ is a [left-stochastic](https://en.wikipedia.org/wiki/Stochastic_matrix), tridiagonal matrix representing the transitions. Starting from a distribution where everyone is in the bottom three leagues, and assuming no one enters or leaves, we observe the existing transition rules push populations towards the higher leagues:

![Alt Text](/assets/images/duolingo_population_evolution.webp)

We can calculate this steady state distribution using linear algebra. In particular, we identify the principal eigenvector of $\mathbf{A}$, which will be associated with an eigenvalue of 1.

A question I have: Can one construct a triadiagonal, left-stochastic matrix $\mathbf{A}$, given a specified principal eigenvector? This would be of use to Duolingo, if say, they wanted most users to be _near_ the top (i.e., in the pearl and obsidian leagues).
