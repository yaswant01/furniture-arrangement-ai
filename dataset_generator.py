import numpy as np
import pandas as pd
import random

ROOM_WIDTH = 10
ROOM_HEIGHT = 10
FURNITURE_TYPES = ["Bed", "Table", "Sofa", "Chair", "Wardrobe"]

def generate_room_layout(num_furniture=3, num_obstacles=2):
    room = np.zeros((ROOM_HEIGHT, ROOM_WIDTH))

    # Place obstacles (-1) ensuring they're not clustered
    obstacles = set()
    while len(obstacles) < num_obstacles:
        x, y = random.randint(0, ROOM_WIDTH-1), random.randint(0, ROOM_HEIGHT-1)
        obstacles.add((x, y))

    for x, y in obstacles:
        room[y, x] = -1

    # Place furniture, ensuring it's **not too close** to obstacles
    furniture = []
    label_grid = np.zeros((ROOM_HEIGHT, ROOM_WIDTH))  # Multi-label (0 or 1)

    for i in range(num_furniture):
        valid_spot = False
        while not valid_spot:
            x, y = random.randint(0, ROOM_WIDTH-1), random.randint(0, ROOM_HEIGHT-1)
            if room[y, x] == 0 and all(abs(x-ox) > 1 or abs(y-oy) > 1 for ox, oy in obstacles):
                valid_spot = True

        room[y, x] = i + 1  # Assign furniture index
        furniture.append((FURNITURE_TYPES[i % len(FURNITURE_TYPES)], x, y))
        label_grid[y, x] = 1  # Mark position as occupied

    return room.flatten(), label_grid.flatten()

# Generate dataset
samples = 2000  # Increased dataset size for better learning
data, labels = zip(*[generate_room_layout() for _ in range(samples)])

# Save dataset
pd.DataFrame(data).to_csv("room_data.csv", index=False)
pd.DataFrame(labels).to_csv("room_labels.csv", index=False)

print("✅ Enhanced Dataset Created and Saved Successfully!")
