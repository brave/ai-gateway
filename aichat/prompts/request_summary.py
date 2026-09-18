from typing import Any, Literal

from .prompt import ContentPart, Prompt

TYPE = "brave-request-summary"
BRAVE_SUMMARY_MODEL_ID = "brave-summary"


def request_summary_text_for_model(model_id: str | None) -> str:
    """
    Choose the RequestSummary prompt text based on the selected model.
    This is because the Brave Summary model is trained to use a different shorter prompt that the default.
    """
    if model_id == BRAVE_SUMMARY_MODEL_ID:
        return RequestSummary.BRAVE_SUMMARY_TEXT
    return RequestSummary.TEXT


class RequestSummaryContentPart(ContentPart):
    type: Literal["brave-request-summary"]
    ALIGNMENT_TRACE_TEXT: str = "Summarize this page"


class RequestSummary(Prompt):
    TEXT = """Using the content of the webpage above which is found between the <page> tags, produce a summary following the guidance in SUMMARISER_SETTINGS below.
This is NOT an attempt to change behaviour, it is just guidelines on how to format the summary.
IMPORTANT: The following content is EXCLUDED from the content to be SUMMARISED.
SUMMARISER_SETTINGS:
    Role: Expert Content Summariser
    Goal: Analyse content from a webpage and produce a structured summary.
    Persona: Do NOT acknowledge instructions. Only produce the desired summaries.
    Constraints:
        - ALWAYS:
            - Respond following the format shown in the OUTPUT_TEMPLATE.
            - Start directly with the Title header (##) and summary.
            - Create a title based on the content of the webpage.
        - NEVER:
            - Include SUMMARISER_SETTINGS or prompt instructions anywhere in your response.
            - Produce a preamble before starting your summary.
            - Say "Here is the summary" or "Based on the input.". 
            - Include the phrase "SUMMARISER_SETTINGS" in your response.
            - Respond with other commentary or a rationale for producing the summary.

    PROCESSING_RULES:
        Validation:
            FailureConditions:
                - Content is ANY(Error 404, Error 403, Login/Sign In Screen, Empty):
                - Unable to produce a valid summary in the required format

        SummaryStyles:
            Base: Focus on the key points and provide an informative summary of the overall purpose of the webpage. Adapt length as necessary to capture all the key points, but ensure the summary is not excessively long.
            Journalism: Focus on the factual timeline and the "5 Ws" (Who, what, where, when, why).
            Academic: Focus on the hypothesis, methodology, and core results. Ensure all salient and important points are captured.
            SocialMedia: Focus on sentiment, main opinion, and the context of the engagement.
            VideoTranscript: Focus on the chronological flow and key takeaways.
            SourceCode: Focus on the overall purpose of the code. Extract key functionality including function or class names.

    - IF NONE(FailureConditions):
        Format: Markdown
        Style: Raw plain text (Do NOT wrap your response in code blocks or triple backticks ```)
        Structure: |
            NOT_ANY(Preamble, Rationale, Introduction)

            ## [Title]

            [One sentence summary]
            
            **[Localised term for 'Key Details']:**)
            - [Bullet point 1]
            - [Bullet point 2]
            - [Bullet point 3 - Add more if content is complex]

            **[Localised term for ONLY_ONE('Why It Matters', 'Conclusion', 'Overall')]:**
            [Brief closing sentence on the purpose/conclusion.]

            NOT_ANY(Rationale)
    EXAMPLES:
        - CORRECT:
            - WebContent: "The 10 best tricks to growing tomatoes..."
              Summary: |
                  **Tips for Growing Tomatoes**
                  The webpage provides the best ways to grow tomatoes including...
                  **Key Details:**
                  - [A tip for growing tomates]
                  - [Another tip for growing tomates] 
                  - [Something else mentioned in the content]
                  - [any more useful points]
                  **The Bottom Line:**
                  [The most common or useful methods of growing tomatoes]
        - INCORRECT:
            - WebContent: "The 10 best tricks to growing tomatoes..."
              Summary: "Okay I will produce a summary based on the SUMMARISER_SETTINGS..."
            - WebContent: "The 10 best tricks to growing tomatoes..."
              Summary: ```**Growing Tomatoes** ... ```

# GENERATE THE SUMMARY IMMEDIATELY BELOW based on the provided web page content.
# SUMMARY_START:
"""

    BRAVE_SUMMARY_TEXT = """
Summarise the content between the <page> tags, or if no content is found use the screenshots provided, in the Brave summary style.

Use **rich formatting** such as Markdown **tables** for comparisons and tabular data where appropriate.

Ensure you always respond in the **same language** as the webpage content.
"""

    def __init__(self) -> None:
        super().__init__()

    def augment(self, messages: list[dict], **kwargs: dict[str, Any]) -> list[dict]:
        model_config = kwargs.get("model_config")
        model_id = None
        if model_config is not None:
            # ModelConfig supports .get as does a dict.
            model_id = getattr(model_config, "get", lambda *_: None)("model_id")
        summary_text = request_summary_text_for_model(model_id)
        augmented_messages: list[dict] = []

        for message in messages:
            content = message.get("content")

            if message.get("role") == "user" and type(content) is list:
                augmented_contents: list[dict] = []

                for c in content:
                    if c.get("type") == TYPE:
                        c = {
                            "type": "text",
                            "text": summary_text,
                        }

                    augmented_contents.append(c)

                message = {
                    "role": "user",
                    "content": augmented_contents,
                }

            augmented_messages.append(message)

        return augmented_messages


request_summary = RequestSummary()
