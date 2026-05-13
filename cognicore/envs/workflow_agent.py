"""
CogniCore Autonomous Workflow Agent Environment.

Agent plans, executes, and recovers from failures in multi-step workflows.
Memory stores successful strategies and failure patterns.
Reflection analyzes bottlenecks and optimizes execution order.

Usage::

    env = cognicore.make("WorkflowAgent-v1", difficulty="medium")
    obs, info = env.reset()
    action = {"task": "validate_data", "retry": False}
    obs, reward, terminated, truncated, info = env.step(action)
"""

from __future__ import annotations

import random
from typing import Any, Dict, List

from cognicore.core.base_env import CogniCoreEnv
from cognicore.core.types import EvalResult


WORKFLOWS = {
    "easy": [
        {
            "id": "data_pipeline",
            "desc": "Execute a data pipeline: ingest → validate → transform → load → verify.",
            "steps": ["ingest", "validate", "transform", "load", "verify"],
            "dependencies": {"validate": ["ingest"], "transform": ["validate"], "load": ["transform"], "verify": ["load"]},
            "failure_probs": {"ingest": 0.1, "validate": 0.15, "transform": 0.1, "load": 0.2, "verify": 0.05},
            "actions": ["ingest", "validate", "transform", "load", "verify", "retry", "skip", "rollback"],
        },
        {
            "id": "deploy_app",
            "desc": "Deploy an application: build → test → stage → approve → deploy.",
            "steps": ["build", "test", "stage", "approve", "deploy"],
            "dependencies": {"test": ["build"], "stage": ["test"], "approve": ["stage"], "deploy": ["approve"]},
            "failure_probs": {"build": 0.15, "test": 0.25, "stage": 0.1, "approve": 0.05, "deploy": 0.2},
            "actions": ["build", "test", "stage", "approve", "deploy", "retry", "rollback", "hotfix"],
        },
        {
            "id": "email_campaign",
            "desc": "Launch email campaign: design → write → review → schedule → send → analyze.",
            "steps": ["design", "write", "review", "schedule", "send", "analyze"],
            "dependencies": {"write": ["design"], "review": ["write"], "schedule": ["review"], "send": ["schedule"], "analyze": ["send"]},
            "failure_probs": {"design": 0.05, "write": 0.1, "review": 0.15, "schedule": 0.05, "send": 0.2, "analyze": 0.05},
            "actions": ["design", "write", "review", "schedule", "send", "analyze", "retry", "skip"],
        },
        {
            "id": "onboarding",
            "desc": "Employee onboarding: create_account → assign_role → setup_workstation → orientation → first_task.",
            "steps": ["create_account", "assign_role", "setup_workstation", "orientation", "first_task"],
            "dependencies": {"assign_role": ["create_account"], "setup_workstation": ["assign_role"], "orientation": ["setup_workstation"], "first_task": ["orientation"]},
            "failure_probs": {"create_account": 0.1, "assign_role": 0.05, "setup_workstation": 0.2, "orientation": 0.05, "first_task": 0.1},
            "actions": ["create_account", "assign_role", "setup_workstation", "orientation", "first_task", "retry", "escalate", "skip"],
        },
    ],
    "medium": [
        {
            "id": "ml_training",
            "desc": "ML training pipeline: collect_data → preprocess → feature_eng → train → evaluate → tune → deploy_model.",
            "steps": ["collect_data", "preprocess", "feature_eng", "train", "evaluate", "tune", "deploy_model"],
            "dependencies": {
                "preprocess": ["collect_data"], "feature_eng": ["preprocess"],
                "train": ["feature_eng"], "evaluate": ["train"],
                "tune": ["evaluate"], "deploy_model": ["tune"]
            },
            "failure_probs": {"collect_data": 0.1, "preprocess": 0.15, "feature_eng": 0.1, "train": 0.3, "evaluate": 0.1, "tune": 0.2, "deploy_model": 0.25},
            "actions": ["collect_data", "preprocess", "feature_eng", "train", "evaluate", "tune", "deploy_model", "retry", "rollback", "checkpoint", "scale_up"],
        },
        {
            "id": "incident_response",
            "desc": "Respond to production incident: detect → triage → diagnose → fix → test_fix → deploy_fix → postmortem.",
            "steps": ["detect", "triage", "diagnose", "fix", "test_fix", "deploy_fix", "postmortem"],
            "dependencies": {
                "triage": ["detect"], "diagnose": ["triage"],
                "fix": ["diagnose"], "test_fix": ["fix"],
                "deploy_fix": ["test_fix"], "postmortem": ["deploy_fix"]
            },
            "failure_probs": {"detect": 0.05, "triage": 0.1, "diagnose": 0.25, "fix": 0.3, "test_fix": 0.2, "deploy_fix": 0.15, "postmortem": 0.05},
            "actions": ["detect", "triage", "diagnose", "fix", "test_fix", "deploy_fix", "postmortem", "retry", "escalate", "rollback", "workaround"],
        },
    ],
    "hard": [
        {
            "id": "microservice_migration",
            "desc": "Migrate monolith to microservices. 10 services, complex dependencies, zero downtime required.",
            "steps": ["analyze_deps", "design_api", "extract_auth", "extract_users", "extract_payments",
                      "extract_notifications", "setup_gateway", "migration_test", "canary_deploy", "full_cutover"],
            "dependencies": {
                "design_api": ["analyze_deps"],
                "extract_auth": ["design_api"], "extract_users": ["design_api"],
                "extract_payments": ["extract_auth", "extract_users"],
                "extract_notifications": ["extract_users"],
                "setup_gateway": ["extract_auth", "extract_payments", "extract_notifications"],
                "migration_test": ["setup_gateway"],
                "canary_deploy": ["migration_test"],
                "full_cutover": ["canary_deploy"],
            },
            "failure_probs": {s: 0.2 + random.random() * 0.15 for s in
                            ["analyze_deps", "design_api", "extract_auth", "extract_users", "extract_payments",
                             "extract_notifications", "setup_gateway", "migration_test", "canary_deploy", "full_cutover"]},
            "actions": ["analyze_deps", "design_api", "extract_auth", "extract_users", "extract_payments",
                       "extract_notifications", "setup_gateway", "migration_test", "canary_deploy", "full_cutover",
                       "retry", "rollback", "checkpoint", "parallel_run", "scale_up", "hotfix"],
        },
    ],
}


class WorkflowAgentEnv(CogniCoreEnv):
    """Autonomous workflow planning and execution environment."""

    def _setup(self, difficulty: str = "easy", **kw):
        self.difficulty = difficulty
        self.workflows = WORKFLOWS.get(difficulty, WORKFLOWS["easy"])
        self.action_space = {"type": "dict", "keys": ["task", "retry"]}
        self.observation_space = {"type": "dict", "keys": [
            "workflow", "completed", "failed", "current_step", "retries_left"
        ]}

    def _generate_tasks(self):
        tasks = []
        for wf in self.workflows:
            tasks.append({
                "workflow": wf["desc"],
                "steps": wf["steps"],
                "dependencies": wf["dependencies"],
                "actions": wf["actions"],
            })
        self._current = self.workflows[0]
        self._completed = set()
        self._failed = set()
        self._retries = 3
        self._step_num = 0
        self._total_steps = len(self._current["steps"])
        return tasks

    def _get_obs(self) -> dict:
        wf = self._current
        # Next available steps (dependencies met)
        available = []
        for s in wf["steps"]:
            if s not in self._completed and s not in self._failed:
                deps = wf["dependencies"].get(s, [])
                if all(d in self._completed for d in deps):
                    available.append(s)
        return {
            "workflow": wf["desc"],
            "completed": list(self._completed),
            "failed": list(self._failed),
            "available": available,
            "retries_left": self._retries,
            "progress": f"{len(self._completed)}/{self._total_steps}",
        }

    def _evaluate(self, action: Any) -> EvalResult:
        wf = self._current
        self._step_num += 1

        if isinstance(action, dict):
            task = action.get("task", "")
            is_retry = action.get("retry", False)
        else:
            task = str(action)
            is_retry = False

        # Handle special actions
        if task == "retry":
            if self._retries > 0 and self._failed:
                self._retries -= 1
                retried = self._failed.pop()
                return EvalResult(base_score=0.5, correct=True,
                    ground_truth="retry failed task", predicted=f"retry {retried}", category="workflow")
            return EvalResult(base_score=0.0, correct=False,
                ground_truth="valid retry", predicted=task, category="workflow")

        if task == "rollback":
            if self._completed:
                rolled = self._completed.pop()
                return EvalResult(base_score=0.3, correct=True,
                    ground_truth="rollback", predicted=f"rollback {rolled}", category="workflow")
            return EvalResult(base_score=0.0, correct=False,
                ground_truth="valid rollback", predicted=task, category="workflow")

        # Check if task is valid
        if task not in wf["steps"]:
            return EvalResult(base_score=0.0, correct=False,
                ground_truth="valid task", predicted=task, category="workflow")

        # Check dependencies
        deps = wf["dependencies"].get(task, [])
        unmet = [d for d in deps if d not in self._completed]
        if unmet:
            return EvalResult(base_score=0.1, correct=False,
                ground_truth=f"complete {unmet} first", predicted=task, category="workflow")

        # Execute with failure probability
        fail_prob = wf["failure_probs"].get(task, 0.1)
        succeeded = random.random() > fail_prob

        if succeeded:
            self._completed.add(task)
            progress = len(self._completed) / self._total_steps
            all_done = len(self._completed) == self._total_steps
            return EvalResult(
                base_score=progress, correct=True,
                ground_truth=task, predicted=f"{task} (success)", category="workflow",
                metadata={"event": "success", "progress": f"{len(self._completed)}/{self._total_steps}", "complete": all_done})
        else:
            self._failed.add(task)
            return EvalResult(
                base_score=0.1, correct=False,
                ground_truth=task, predicted=f"{task} (failed)", category="workflow",
                metadata={"event": "failure", "fail_prob": fail_prob})
