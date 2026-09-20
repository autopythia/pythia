import argparse

from pythia.interaction import ChatCompletionsEndpoint
from pythia.interaction import ChatCompletionsModel
from pythia.interaction import BUILTIN_MODEL_CATALOG
from pythia.interaction import Message
from pythia.interaction import InteractionContext
from pythia.interaction import UserInteraction


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one caller-controlled Chat Completions sample.",
    )
    parser.add_argument(
        "--endpoint-url",
        default="http://127.0.0.1:8000/v1/chat/completions",
    )
    parser.add_argument("--model")
    parser.add_argument("--endpoint-api-key", default=None)
    parser.add_argument("prompt")
    args = parser.parse_args()

    endpoint = ChatCompletionsEndpoint(
        binding=BUILTIN_MODEL_CATALOG.bind(
            "chat-completions",
            args.model,
            endpoint_url=args.endpoint_url,
            endpoint_auth=(
                "supplied" if args.endpoint_api_key is not None else "none"
            ),
        ),
        api_key=args.endpoint_api_key,
    )
    model = ChatCompletionsModel(endpoint=endpoint)
    context = InteractionContext(
        [
            Message(
                role="system",
                content="You are a concise and helpful assistant.",
            ),
        ]
    )
    user_interaction = UserInteraction(
        items=(Message(role="user", content=args.prompt),),
    )
    context.extend(user_interaction.context_items())

    sample = model.sample(context)
    context.extend(sample.context_items())

    if sample.last_assistant_text is not None:
        print(sample.last_assistant_text)


if __name__ == "__main__":
    main()
