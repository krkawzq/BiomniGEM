system_base = """You are a helpful biology expert. {task_abstract}

## Task
{task_description_list}
"""

user_base = """{task}

You are given the correct answer:
{answer}

---

Please explain **why** this answer is correct by providing a clear and detailed reasoning process
before stating the final conclusion.

Follow these requirements carefully:
- You must reason using biological knowledge and domain understanding.
- Each reasoning step must have clear biological meaning and causal coherence.
- You must not reveal or imply that you already know the final answer; reasoning should unfold naturally.
- Each step should logically follow from the previous one, forming a consistent causal chain.
- Present your final reasoning and conclusion in a well-formatted JSON block as shown below:

```json
{{
    "traces": [
        "...", 
        "...", 
        "..."
    ],
    "conclusion": "..."
}}
"""
