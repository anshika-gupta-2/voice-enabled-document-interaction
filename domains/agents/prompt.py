ORCHESTRATOR_PROMPT = """You are a helpful AI assistant, collaborating with other assistants.
Use the provided tools to progress towards answering the question.

If you are unable to fully answer, that's OK, another assistant with different tools will help where you left off.
Execute what you can to make progress.
If you or any of the other assistants have the final answer or deliverable, prefix your response with FINAL ANSWER so the team knows to stop.
{suffix}"""


DOC_PARSER_PROMPT="""
Extract key content from the following document and structure it into a well-organized format:"

{context}
"""

DISTILL_SUMMARY_PROMPT="""
The following is a set of summaries: \n{docs}

Take these and distill it into a final, consolidated summary of the main themes.
"""