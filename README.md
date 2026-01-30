# MiniRTS Reinforcement Learning Project

## Without Training

https://github.com/user-attachments/assets/45602c61-f4fb-42e7-9a99-6dd74c42020b

## With Training

https://github.com/user-attachments/assets/a332d362-9cef-47f3-b480-28025a0f309e

This is a passion project aimed at honing my skills, with the long term goal of reproducing something conceptually similar to [AlphaStar](https://en.wikipedia.org/wiki/AlphaStar_(software)). To work toward this, I built a custom environment for a miniature RTS game and applied reinforcement learning techniques to train agents within it.

The game is currently very simple. A Town Center produces villagers, which can move around the map, gather gold, and return it to buildings. Villagers can also convert into additional Town Centers or Barracks. Barracks can produce troops, which are able to deal damage to enemy units.

The reinforcement learning algorithm currently implemented is Advantage Actor Critic (A2C). Each tile of the map is one hot encoded into roughly a dozen feature channels and passed into the neural network. The policy is trained through self play. The opponent is only updated to the new policy once it achieves a significantly higher win rate compared to the previous version.

As shown above, without training the villagers wander around aimlessly. After training, the behavior of collecting resources and producing additional villagers becomes much more apparent.

## TODO

- Randomize the spawn locations of various game objects so the policy does not overfit to the map
- Optimize the game logic so training runs faster
- Update the reward function to prioritize winning the game through combat rather than maximizing villager production
- Implement PPO and compare its performance
- Add multiple troop types with a rock paper scissors style counter system, then implement a multi agent learning setup inspired by the AlphaStar paper so the agent learns generalized strategies rather than overfitting to past versions of itself
- Add fog of war so agents no longer have perfect information
