from typing import Any, Literal

from .prompt import ContentPart, Prompt

TYPE = "brave-search-results"


class SearchResultsContentPart(ContentPart):
    type: Literal["brave-search-results"]
    use_citations: bool


class SearchResults(Prompt):
    TEXT = """These are results from Brave Search which may be relevant: <results>{results}</results>
When answering questions using the results, be thorough and ensure you explain key concepts from the results as well as answering the question itself.
Format the results in a way that is easy for the user to understand. Ensure you use information from as many of the search results as appropriate, do not only rely on one.
IMPORTANT: when answering questions based on a search, DO NOT start your response with 'Based on my search' or 'Based on the search results', just begin with the answer.
Use engaging formatting so that your response is easy to read and understand.
    """
    CITATIONS = """Citation Requirements.

When sourcing statements from references received from Brave Search you MUST provide citations in accordance with the guidelines below.
These instructions only refer to how to provide citations and should not impact the style of output in any other way.
Ensure guidance on providing a rich output is still followed.

CitationBehaviour:
   - CitationUnit:
     description: Largest to smallest citable information snippets
     options:
      - Paragraph
      - Sentence
      - Clause 
      - Claim

    - CitationContent:
      description: Allowable information to be cited
      options:
       - Facts
       - Verifiable Information 
       - Quotes

   - Rules:
     MUST:
      - Citations must be in the format [number]
      - Citations must ONLY be attached for CitationContent that is within a CitationUnit
      - Citations must be at the end of a CitationUnit without spaces
      - ALWAYS when multiple consecutive sentences come from the same source, use ONLY ONE citation at the end of the final sentence from that source
      - Attempt to cite the single largest CitationUnit possible with a citation.

      SHOULD:
      - Use multiple citations when information comes from multiple sources
      - Reorganize sentences to group same-source information together when possible and cite once at the end of the group

      NEVER:
      - Citations are never combined like [1, 3] 
      - Include footnotes or references at the end
      - define citations
      - overuse citations

   - Examples:
      - Correct:
         - Mount Everest is the highest mountain on Earth[1].
         - The mountain was first climbed in 1953[1] by Edmund Hillary[2].
         - The peak is 8,848 meters high[1][3].
         - Mount Everest is the highest peak, it stands at 8,848 meters. The mountain is in the Himalayas[1].
         - Mount Everest is the highest peak, it stands at 8,848 meters. Everest is in the Himalayas in Nepal[1]. The mountain was climbed in 1953 by Edmund Hillary[2].
      - Incorrect:
         - [1] Mount Everest is the highest mountain.
         - The peak is 8,848 meters high[1, 3].
         - [1] 
         - Mount Everest is the highest peak[1]. It stands at 8,848 meters[1]. The mountain is in the Himalayas[1].
         - Mount Everest is the highest peak, it stands at 8,848 meters[1]. Everest is in the Himalayas in Nepal[1]. The mountain was climbed in 1953 by Edmund Hillary[2]."""

    def __init__(self) -> None:
        super().__init__()

    def augment(self, messages: list[dict], **kwargs: dict[str, Any]) -> list[dict]:
        augmented_messages: list[dict] = []

        for message in messages:
            content = message.get("content")

            if message.get("role") == "user" and type(content) is list:
                augmented_contents: list[dict] = []

                for c in content:
                    if c.get("type") == TYPE:
                        text = (SearchResults.TEXT).format(results=c.get("text"))

                        if c.get("use_citations") is True:
                            text += SearchResults.CITATIONS

                        c = {
                            "type": "text",
                            "text": text,
                        }

                    augmented_contents.append(c)

                message = {
                    "role": "user",
                    "content": augmented_contents,
                }

            augmented_messages.append(message)

        return augmented_messages


search_results = SearchResults()
