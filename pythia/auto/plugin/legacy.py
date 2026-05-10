import asyncio
from dataclasses import dataclass

from pythia.auto.plugin import AutopythiaPlugin


@dataclass
class Legacy(AutopythiaPlugin):
    @staticmethod
    def _complete_immediately(self, step_ctr: int | None) -> None:
        from pythia.auto.kernel import EndControlEvent, StartControlEvent

        if step_ctr is None:
            step_ctr = self._fresh_step_ctr(self._session)

        self._enqueue_event(StartControlEvent(step_ctr))
        self._enqueue_event(EndControlEvent(step_ctr))

    @staticmethod
    async def auto(self, step_ctr: int, query: str = ""):
        await self.init(step_ctr, query)

    pythia = auto

    @staticmethod
    async def qq(self, step_ctr: int, query: str = ""):
        from pythia.auto.kernel import (
            AnswerOutputEvent,
            AtomicOutputEvent,
            BasicOutputEvent,
            EndControlEvent,
            StartControlEvent,
            ThinkingOutputEvent,
        )
        from pythia.clock import Timestamp
        from pythia.extract import Message
        from pythia.term_utils import green

        think_model_path = self.think_model
        think_model = self.services.registry.find_model(think_model_path)
        think_sampling_params = {
            "max_tokens": 262144,
            "temperature": 1.0,
        }

        if step_ctr is None:
            step_ctr = self._fresh_step_ctr(self._session)

        self._enqueue_event(StartControlEvent(step_ctr))

        qq_query = [
            {
                "role": "user",
                "content": query,
            },
        ]
        self.append_transcript(self._session, query)

        self._enqueue_event(BasicOutputEvent(green("Thinking...", bold=True)))

        t0 = Timestamp()
        qq_result = self.services.client.message(
            think_model,
            qq_query,
            think_sampling_params,
            fresh=True,
        )
        self.append_history(self._session, step_ctr, messages=qq_query, t0=t0)
        qq_result = await qq_result
        t1 = Timestamp()

        res_event = AtomicOutputEvent()
        res_event.append(BasicOutputEvent(green(f"Thought for {(t1 - t0).pretty_format()}", bold=True)))

        message = qq_result.message()
        thinking_part = Message.get_thinking_part(message)
        answer = Message.get_text(message)

        if thinking_part:
            event = ThinkingOutputEvent(thinking_part["thinking"])
            self.append_transcript(self._session, event)
            res_event.append(event)

        event = AnswerOutputEvent(answer)
        self.append_transcript(self._session, event)
        res_event.append(event)
        self._enqueue_event(res_event)

        self._enqueue_event(EndControlEvent(step_ctr))

    @staticmethod
    async def cleanhtml(self, step_ctr: int, query: str):
        from pythia.auto.kernel import (
            AnswerOutputEvent,
            AtomicOutputEvent,
            BasicOutputEvent,
            EndControlEvent,
            StartControlEvent,
            ThinkingOutputEvent,
        )
        from pythia.clock import Timestamp
        from pythia.extract import Message
        from pythia.term_utils import green

        think_model_path = self.think_off_model
        think_model = self.services.registry.find_model(think_model_path)
        think_sampling_params = {
            "max_tokens": 65536,
            "temperature": 1.0,
        }

        input_path = query.strip()
        input_file = open(input_path, "r")

        target_chunk_size = 5000
        chunks = []
        chunk_len = 0
        chunk = []
        for line in input_file:
            chunk_len += len(line)
            chunk.append(line)
            if chunk_len >= target_chunk_size:
                chunks.append("".join(chunk))
                chunk_len = 0
                chunk = []
        if chunk:
            chunks.append("".join(chunk))

        input_file.close()

        input_chunks = chunks
        output_chunks = []

        if step_ctr is None:
            step_ctr = self._fresh_step_ctr(self._session)

        self._enqueue_event(StartControlEvent(step_ctr))
        self._enqueue_event(BasicOutputEvent(green(f"Computed {len(input_chunks)} chunks", bold=True)))
        self._enqueue_event(BasicOutputEvent(green("Thinking...", bold=True)))

        chunk_tasks = []
        chunk_outputs = {}

        async def _pair(key, fut):
            return key, await fut

        def create_task_pair(key, fut):
            return asyncio.create_task(_pair(key, fut))

        total_t0 = Timestamp()

        for chunk_idx, chunk in enumerate(input_chunks):
            chunk_query = [
                {
                    "role": "system",
                    "content": (
"""You are a meticulous data analyst. Below, you will be given a section of HTML to clean for algorithmic processing. Broadly, the intention of this HTML cleaning is to preserve the structure and content of the HTML document, while removing unneeded style and format.

You should follow these specific HTML cleaning criteria:

- Remove `class`, `style`, and non-rendered metadata fields inside tags.
- Remove `link`, `script`, and `style` blocks.
- Remove leading spaces or tabs that are not rendered (e.g. not inside a pre block).
- Do not add any delimiters (e.g. fenced code block) around your HTML output.

Your output should consist of and only of the cleaned HTML corresponding to the user input.
"""
                    ),
                },
                {
                    "role": "user",
                    "content": chunk,
                },
            ]

            t0 = Timestamp()
            chunk_result = self.services.client.message(
                think_model,
                chunk_query,
                think_sampling_params,
                fresh=True,
            )
            chunk_tasks.append(create_task_pair(chunk_idx, chunk_result))
            self.append_history(self._session, step_ctr, messages=chunk_query, t0=t0)

        async def _wait_tasks(tasks):
            rem_tasks = {task for task in tasks}
            while rem_tasks:
                done, _pending = await asyncio.wait(rem_tasks, return_when=asyncio.FIRST_COMPLETED)
                rem_tasks -= done
                for done_task in done:
                    yield done_task.result()

        async for chunk_idx, chunk_result in _wait_tasks(chunk_tasks):
            t1 = Timestamp()

            res_event = AtomicOutputEvent()
            res_event.append(
                BasicOutputEvent(green(f"Thought for {(t1 - total_t0).pretty_format()}", bold=True))
            )

            message = chunk_result.message()
            thinking_part = Message.get_thinking_part(message)
            answer = Message.get_text(message)

            output_text = answer
            if not output_text.endswith("\n"):
                output_text = f"{output_text}\n"

            with open(f"_tmp_clean.{chunk_idx}.html", "w") as output_file:
                print(output_text, end="", file=output_file, flush=True)

            chunk_outputs[chunk_idx] = output_text

            if thinking_part:
                event = ThinkingOutputEvent(thinking_part["thinking"])
                self.append_transcript(self._session, event)
                res_event.append(event)

            event = AnswerOutputEvent(answer)
            res_event.append(event)
            self._enqueue_event(res_event)

        self._enqueue_event(EndControlEvent(step_ctr))

        for chunk_idx in range(len(input_chunks)):
            output_chunks.append(chunk_outputs[chunk_idx])

        output_text = "".join(output_chunks)
        with open("_tmp_clean.html", "w") as output_file:
            print(output_text, end="", file=output_file, flush=True)

    @staticmethod
    async def a(self, step_ctr: int, query: str = ""):
        del query
        Legacy._complete_immediately(self, step_ctr)

    accept = a

    @staticmethod
    async def status(self, step_ctr: int, query: str = ""):
        del query
        Legacy._complete_immediately(self, step_ctr)

    @staticmethod
    async def revise(self, step_ctr: int, query: str = ""):
        del query
        Legacy._complete_immediately(self, step_ctr)

    @staticmethod
    async def review(self, step_ctr: int, query: str = ""):
        del query
        Legacy._complete_immediately(self, step_ctr)
