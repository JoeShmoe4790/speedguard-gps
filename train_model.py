"""
train_model.py
Run this once to generate model.pkl before launching the app.
"""
import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingClassifier

np.random.seed(42)
n = 5000

speed_over  = np.random.uniform(0, 40, n)
hour        = np.random.randint(0, 24, n)
is_highway  = np.random.randint(0, 2, n)
speed_limit = np.random.choice([55, 60, 65, 70, 75], n)

# Simulate realistic ticket probability
def ticket_prob(over, hr, hw, lim):
    p = 0.03
    p += over * 0.040
    p += max(0, over - 15) * 0.055
    if hr in range(7, 10) or hr in range(16, 19):
        p += 0.14
    elif hr >= 22 or hr <= 5:
        p += 0.07
    if not hw:
        p *= 1.15
    if lim <= 55:
        p *= 1.20
    return min(p, 0.97)

probs  = np.array([ticket_prob(o, h, hw, l)
                   for o, h, hw, l in zip(speed_over, hour, is_highway, speed_limit)])
labels = np.random.binomial(1, probs)

X = np.column_stack([speed_over, hour, is_highway, speed_limit])
model = GradientBoostingClassifier(n_estimators=150, max_depth=4, random_state=42)
model.fit(X, labels)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅  model.pkl saved  —  accuracy:",
      round((model.predict(X) == labels).mean(), 3))
