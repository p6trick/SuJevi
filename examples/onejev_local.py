from sujevi import LocalOneJev
from qev import Choice, Noul, Score


model = LocalOneJev()
response, meta = model.system_one(
    state={
        "task": "Send the completed report to Alice",
        "status": "The report is complete but has not been sent.",
    },
    questions={
        "done": Noul("The report has been sent to Alice"),
        "next": Choice(
            "What should happen next?",
            {"send": "send the report to Alice", "stop": "stop without sending"},
        ),
        "progress": Score(
            "How far along is the task?",
            ["not started", "in progress", "almost done", "complete"],
        ),
    },
)

print(response.model_dump_json(indent=2))
print(meta)
