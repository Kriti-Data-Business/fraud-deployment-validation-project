"""Part C: an LLM-style fraud-triage assistant and the harness that validates it.

The assistant never decides. It turns a flagged application's structured facts
into a short, cited case summary for a human investigator. Everything in this
package exists to answer one question: how do we know the assistant's output
can be relied on, and what happens when it cannot?
"""
