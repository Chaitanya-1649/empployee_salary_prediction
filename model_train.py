import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

data = pd.read_csv('adult.csv')
data = data.dropna()

# Encode categorical columns
data['workclass'] = data['workclass'].astype('category').cat.codes
data['education'] = data['education'].astype('category').cat.codes
data['occupation'] = data['occupation'].astype('category').cat.codes

# Add synthetic experience column (random for now)
data['experience'] = np.random.randint(0, 30, size=len(data))

# Features including the synthetic experience
X = data[['age', 'education', 'occupation', 'hours-per-week', 'experience']]
y = data['income'].apply(lambda x: 1 if '>50K' in x else 0)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
