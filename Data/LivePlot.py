import matplotlib.pyplot as plt
from collections import deque
from threading import Lock
import time

# Shared data and lock
actor_losses = deque(maxlen=100)
critic_losses = deque(maxlen=100)
total_losses = deque(maxlen=100)
plot_data_lock = Lock()

def live_plot(update_interval=0.1):
    """
    Live plot function for PPO training losses.
    :param update_interval: Time in seconds between plot updates.
    """
    try:
        plt.ion()  # Enable interactive mode
        fig, ax = plt.subplots()
        while True:
            with plot_data_lock:  # Ensure thread safety
                ax.clear()
                ax.plot(actor_losses, label="Actor Loss", color="blue")
                ax.plot(critic_losses, label="Critic Loss", color="green")
                ax.plot(total_losses, label="Total Loss", color="red")
            ax.set_title("Live PPO Training Losses")
            ax.set_xlabel("Batch")
            ax.set_ylabel("Loss")
            ax.legend()
            plt.pause(0.01)  # Pause to refresh the plot
            time.sleep(update_interval)  # Control update frequency
    except KeyboardInterrupt:
        print("Live plot terminated.")
    finally:
        plt.close(fig)  # Close the plot when done
