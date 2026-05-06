import numpy as np  # importing numpy for numerical operations

# Generate a synthetic gameplay session
np.random.seed(42) #for reproducibility

# Base score: gradual increase over 1000 timepoints
n_timepoints = 1000 # number of timepoints in the session
base_score = np.cumsum(np.random.randint(1, 5, size=n_timepoints)) # cumulative sum to create a gradual increase in score

# Add 15 reward spikes at random positions
reward_positions = np.sort(np.random.choice(n_timepoints, size=15, replace=False)) # random positions for rewards, sorted for better visualization
reward_magnitudes = np.random.randint(50, 200, size=15) # random magnitudes for rewards
base_score[reward_positions] += reward_magnitudes # add rewards to the base score at the specified positions

# Calculate rolling mean over a window of 50
window = 50 # size of the rolling window
rolling_mean = np.convolve(base_score, np.ones(window)/window, mode='valid') # calculate rolling mean using convolution

# Print summary stats
print(f"Session length: {n_timepoints} timepoints")
print(f"Number of rewards: {len(reward_positions)}")
print(f"Reward positions: {reward_positions}")
print(f"Average reward magnitude: {reward_magnitudes.mean():.1f}")
print(f"Final score: {base_score[-1]}")
print(f"Score std: {base_score.std():.1f}")
# Plotting (optional, requires matplotlib)
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
plt.plot(base_score, label='Score')
plt.plot(np.arange(window-1, n_timepoints), rolling_mean, label='Rolling Mean (window=50)', color='orange')
plt.scatter(reward_positions, base_score[reward_positions], color='red', label='Rewards', zorder=5)
plt.title('Simulated Gameplay Session')
plt.xlabel('Timepoints')
plt.ylabel('Score')
plt.legend()
plt.grid()
plt.show()
