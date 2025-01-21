# Tim's Financial PPO Model

### This is a custom PPO model setup with the express intent to enhance Financial Portfolio stock allocations 

The Environment used is my own custom one with generic data at the moment though I plan on adding better input features 
with better, more descriptive qualities later

The PPO framework is for actor-critic analysis with continuous learning in mind as with most PPO based implementations
- More specifically my Actor model for deriving the policy is a GRU based model in order to better capture temporal relationships
  - One note about this, it may be better to use this model with more general feature inputs as the nitty-gritty of the stock market is very noisy
- And my critic is also a gru model, but obviously it only outputs the estimated value of the current state
- Right now the reward isn't balanced well enough per each state and learning either stagnates or stops
  - To rectify this  I am thinking about gather things like sentiment for easier decision-making, but also making the reward based on immediate change to start
  - but then transitioning to a more refined model once the short term is at least circumvented and we can make progress