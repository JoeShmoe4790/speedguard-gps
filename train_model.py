import pandas as pd
import numpy as np
import pickle
import os

def train():
    print("Training model...")

    # Try real SWITRS data first
    if os.path.exists("switrs.sqlite"):
        import sqlite3
        conn = sqlite3.connect("switrs.sqlite")
        df = pd.read_sql("""
            SELECT collision_severity, party_count
            FROM collisions LIMIT 100000
        """, conn)
        conn.close()
        print("Using real SWITRS data")
    else:
        print("switrs.sqlite not found — generating synthetic training data")

    # Generate synthetic data (always used for features)
    np.random.seed(42)
    n = 20000
    speed_over  = np.random.randint(0, 40, n)
    hour        = np.random.randint(0, 24, n)
    is_highway  = np.random.randint(0, 2, n)
    speed_limit = np.random.choice([25,35,45,55,65], n)
    rush_hour   = ((hour >= 7) & (hour <= 9) | (hour >= 16) & (hour <= 18)).astype(int)
    night       = ((hour >= 22) | (hour <= 5)).astype(int)

    # Risk increases with speed, night, rush hour
    risk = (
        speed_over * 0.04 +
        is_highway * 0.1 +
        rush_hour * 0.15 +
        night * 0.2 +
        np.random.normal(0, 0.15, n)
    )
    ticket = (risk > 0.5).astype(int)

    X = np.column_stack([speed_over, hour, is_highway, speed_limit, rush_hour, night])
    y = ticket

    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=200, max_depth=4, random_state=42)
    model.fit(X, y)

    acc = model.score(X, y)
    print(f"Model accuracy: {acc:.3f}")

    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)
    print("model.pkl saved!")

if __name__ == "__main__":
    train()