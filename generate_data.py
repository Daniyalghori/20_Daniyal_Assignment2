import pandas as pd
import random

positive = ["Amazing experience on the new Mumbai Coastal Road!", "The coastal road is a masterclass in civil engineering.", "Traffic in Worli has reduced significantly.", "Brilliant infrastructure and sea views.", "Finally, Mumbai gets the infrastructure it deserves."]
negative = ["The coastal road entry points are a traffic nightmare.", "Potholes already appearing on connecting roads.", "Why is the coastal road closed on weekends?", "Still stuck in traffic at Haji Ali.", "Too much congestion at the toll plaza."]
neutral  = ["Phase 2 of the Mumbai Coastal Road opens next month.", "Taking the coastal road today for my commute.", "The speed limit on the coastal road is 80 km/h.", "News: Coastal road connects to Bandra-Worli Sea Link.", "Just saw the new tunnels on the coastal road."]

data = []
for i in range(100):
    if i < 40: data.append([positive[i%5] + f" [{i}]", "positive"])
    elif i < 70: data.append([negative[i%5] + f" [{i}]", "negative"])
    else: data.append([neutral[i%5] + f" [{i}]", "neutral"])

random.shuffle(data)
df = pd.DataFrame(data, columns=["Tweet", "Sentiment"])
df.to_csv("dataset.csv", index=False)
print("Success: dataset.csv created with 100 tweets!")