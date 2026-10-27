from sujevi import LocalJevOmni


model = LocalJevOmni()
result = model.predict(
    state="The meeting starts at 10 AM. It is now 9 AM.",
    question="Has the meeting started?",
    options=["Yes", "No"],
)
print(result)
