import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder

data = [
    ["python programming", "technology", "anlytical", "Software Engineer"],
    ["java", "technology", "anlytical", "Software Engineer"],
    ["biology", "healthcare", "helping", "Doctor"],
    ["chemistry", "healthcare", "anlytical", "Pharmacist"],
    ["math", "finance", "anlytical", "Data Analyst"],
    ["math", "technology", "anlytical", "AI Engineer"],
    ["design", "creativity",  "artistc", "Graphic Designer"],
    ["drawing", "creativity",  "artistc", "Animator"],
    ["teaching", "education", "social", "Teacher"],
    ["communication", "education", "social", "Counselor"],
    ["writing", "media", "creative", "Content Writer"],
    ["journalism", "media", "social", "Journalist"],
    ["sports", "fitness", "active", "Fitness trainer"],
    ["yoga", "fitness", "calm", "Yoga instructor"],
    ["business", "management", "leader", "Entrepreneur"],
    ["management", "finance", "leader", "Manager"]
]

df = pd.DataFrame(data, columns=["skils", "interest", "personality", "career"])

encoder = OneHotEncoder(handle_unknown='ignore')
X = encoder.fit_transform(df[["skils", "interest", "personality"]])

y = df["career"]

model = DecisionTreeClassifier()
model.fit(X, y)

print("\n---------- AI Career Recommendation System ----------\n")

print("Let's Find Your Perfect Career:\n")

print("----- Available Skills -----")
print(", ".join(sorted(df["skils"].unique())))

print("\n----- Available Interests -----")
print(", ".join(sorted(df["interest"].unique())))

print("\n----- Available Personalities -----")
print(", ".join(sorted(df["personality"].unique())))

skils = input("\nEnter your skill: ").strip().lower()
interest = input("Enter your interest: ").strip().lower()
personality = input("Enter your personlity: ").strip().lower()


input_df = pd.DataFrame([[skils, interest, personality]],
                        columns=["skils", "interest", "personality"])

input_encoded = encoder.transform(input_df)

prediction = model.predict(input_encoded)[0]

print("\n-----Recommended Career-----\n",prediction)
