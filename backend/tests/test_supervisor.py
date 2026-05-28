"""Supervisor agent tests — Plan/PlanStep schema validation.

Pure tests: don't hit the LLM. They verify the structured-output schema
that the planner LLM must return.
"""

import pytest
from pydantic import ValidationError

from agents.supervisor_P import Plan, PlanStep


def test_plan_step_accepts_valid_agent():
    step = PlanStep(agent="sql", task="Count active customers")
    assert step.agent == "sql"
    assert step.task == "Count active customers"


def test_plan_step_rejects_unknown_agent():
    with pytest.raises(ValidationError):
        PlanStep(agent="not_a_real_agent", task="anything")


def test_plan_step_requires_task():
    with pytest.raises(ValidationError):
        PlanStep(agent="sql")   # missing task


def test_plan_accepts_empty_steps():
    """Empty steps is valid — used when the supervisor refuses an off-topic query."""
    plan = Plan(rationale="Off-topic", steps=[])
    assert plan.steps == []


def test_plan_accepts_multiple_steps():
    plan = Plan(
        rationale="forecast + market context in parallel",
        steps=[
            PlanStep(agent="forecast", task="Forecast Ladieswear revenue"),
            PlanStep(agent="web", task="2026 womenswear outlook"),
        ],
    )
    assert len(plan.steps) == 2
    assert {s.agent for s in plan.steps} == {"forecast", "web"}
