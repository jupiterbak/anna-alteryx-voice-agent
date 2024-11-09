from llama_index.core.workflow import Event
from llama_index.llms.openai import OpenAI

from llama_index.core.workflow import (
    Workflow,
    StartEvent,
    StopEvent,
    Context,
    step,
)


class TestingDone(Event):
    output: str
    user_transcript: str

class ValidationErrorEvent(Event):
    error: str
    wrong_output: str
    user_transcript: str


TESTING_PROMPT = """As a grader, your task is to evaluate if the user transcript is a question.

    User transcript:
    -------------------
    {user_transcript}

    Evaluation Criteria:
    - Consider if the transcript is a question by checking if the user is asking for information about a specific topic.
    - The evaluation should not be overly stringent; the primary objective is to identify and filter out clearly irrelevant retrievals.

    Decision:
    - Assign a binary score to indicate if the transcript is a question.
    - Use 'yes' if the transcript is a question, or 'no' if it is not.

    Please provide your binary score ('yes' or 'no') below to indicate if the transcript is a question."""

REFLECTION_PROMPT = """
    You already created this output previously:
    ---------------------
    {wrong_answer}
    ---------------------

    This caused the following error: {error}

    Try again, the response must contain only 'yes' or 'no' without any other text.
    """

class TestRelevanceWorkflow(Workflow):
    max_retries: int = 3

    @step
    async def test_relevance(
        self, ctx: Context, ev: StartEvent | ValidationErrorEvent
    ) -> StopEvent | TestingDone:
        current_retries = await ctx.get("retries", default=0)
        if current_retries >= self.max_retries:
            return StopEvent(result="no") # per default user is not asking a question
        else:
            await ctx.set("retries", current_retries + 1)

        if isinstance(ev, StartEvent):
            user_transcript = ev.get("user_transcript")
            if user_transcript is None:
                return StopEvent(result="no, please provide some text in input")
            reflection_prompt = ""
        elif isinstance(ev, ValidationErrorEvent):
            user_transcript = ev.user_transcript
            reflection_prompt = REFLECTION_PROMPT.format(
                error=ev.error, wrong_answer=ev.wrong_output
            )
        
        testing_prompt = TESTING_PROMPT.format(user_transcript=user_transcript)
        if reflection_prompt:
            testing_prompt += reflection_prompt

        llm = OpenAI(model="gpt-4o", api_key=ev.get("openai_apikey"))
        output = await llm.acomplete(testing_prompt)

        return TestingDone(output=str(output), user_transcript=user_transcript)

    @step
    async def validate(
        self, ev: TestingDone
    ) -> StopEvent | ValidationErrorEvent:
        if ev.output not in ["yes", "no"]:
            return ValidationErrorEvent(
                error="Invalid output", wrong_output=ev.output, user_transcript=ev.user_transcript
            )

        return StopEvent(result=ev.output)