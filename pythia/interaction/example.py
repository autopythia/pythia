import argparse

from pythia.interaction import ChatCompletionsEndpoint
from pythia.interaction import ChatCompletionsModel
from pythia.interaction import Message
from pythia.interaction import ModelContext
from pythia.interaction import UserInteraction


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one caller-controlled Chat Completions sample.",
    )
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("prompt")
    args = parser.parse_args()

    endpoint = ChatCompletionsEndpoint(
        api_url=args.api_url,
        model=args.model,
        api_key=args.api_key,
    )
    model = ChatCompletionsModel(endpoint=endpoint)
    context = ModelContext(
        [
            Message(
                role="system",
                text="You are a concise and helpful assistant.",
            ),
        ]
    )
    user_interaction = UserInteraction(
        items=(Message(role="user", text=args.prompt),),
    )
    context.extend(user_interaction.context_items())

    sample = model.sample(context)
    context.extend(sample.context_items())

    if sample.last_assistant_text is not None:
        print(sample.last_assistant_text)


if __name__ == "__main__":
    main()
