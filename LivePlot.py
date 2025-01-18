import matplotlib.pyplot as plt
from collections import deque
import time

# Global shared deque for live data
actor_losses = deque(maxlen=100)
critic_losses = deque(maxlen=100)
total_losses = deque(maxlen=100)

# Function to update the live plot
def live_plot():
    plt.ion()  # Interactive mode
    fig, ax = plt.subplots()
    while True:
        ax.clear()
        ax.plot(actor_losses, label="Actor Loss", color="blue")
        ax.plot(critic_losses, label="Critic Loss", color="green")
        ax.plot(total_losses, label="Total Loss", color="red")
        ax.set_title("Live Training Losses")
        ax.set_xlabel("Batch")
        ax.set_ylabel("Loss")
        ax.legend()
        plt.pause(0.1)  # Pause to refresh the plot
        time.sleep(0.1)  # Adjust for efficiency
