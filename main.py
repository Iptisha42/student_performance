
import pandas as pd

df = pd.read_csv("student_performance.csv")

print("First 5 rows:")
print(df.head())
print("\nDataset info:")
print(df.info())
print("\nSummary statistics:")
print(df.describe())
correlation = df["Hours_Studied"].corr(df["Final_Score"])
print(f"\nCorrelation between Hours Studied and Final Score: {correlation}")

import matplotlib.pyplot as plt

plt.scatter(df["Hours_Studied"], df["Final_Score"])
plt.xlabel("Hours Studied")
plt.ylabel("Final Score")
plt.title("Hours Studied vs Final Score")
plt.show()

X = df[["Hours_Studied", "Previous_Score", "Sleep_Hours", "Attendance_Percent"]]
y = df["Final_Score"]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining rows: {len(X_train)}, Testing rows: {len(X_test)}")
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)  
from sklearn.metrics import mean_absolute_error, r2_score

predictions = model.predict(X_test)

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"\nMean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.2f}")
print("\n--- Predict Your Own Score ---")
hours = float(input("Hours studied: "))
prev_score = float(input("Previous score: "))
sleep = float(input("Sleep hours: "))
attendance = float(input("Attendance percent: "))

user_data = pd.DataFrame([{
    "Hours_Studied": hours,
    "Previous_Score": prev_score,
    "Sleep_Hours": sleep,
    "Attendance_Percent": attendance
}])

predicted_score = model.predict(user_data)[0]
print(f"\nPredicted Final Score: {predicted_score:.2f}")

if predicted_score >= 75:
    print("That looks like a good score!")
else:
    print("There's room to improve — try adjusting study hours or attendance.")