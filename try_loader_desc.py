from tqdm import tqdm
import time

loader = tqdm(range(60))
for i in loader:
    time.sleep(1)
    loader.set_description(f"iteration {i}; mse: {i*0.2}")


