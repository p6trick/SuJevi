"""Run a System One JSON request directly against local OneJev."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sujevi import LocalOneJev


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("request", help="System One request JSON")
    parser.add_argument("--model", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--debias", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    body = json.loads(Path(args.request).read_text(encoding="utf-8"))
    kwargs = {"device": args.device}
    if args.model:
        kwargs["model_path"] = args.model
    model = LocalOneJev(**kwargs)
    response, meta = model.system_one(
        body.get("state"),
        body["questions"],
        media=body.get("media"),
        debias=args.debias,
        debug=args.debug,
    )
    output = response.model_dump(mode="json")
    output["qev"] = meta
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
