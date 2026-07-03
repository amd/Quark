# Unified Interaction Contract

All user-facing Quark skills must follow this five-stage skeleton.

## 1. Intake

- identify the user goal in plain language
- collect only the missing information required to continue
- detect whether the user wants planning only or plan plus execution

## 2. Route

- decide whether the request belongs to an atomic skill or a workflow
- explain the selected path in one short sentence
- if the request spans multiple concerns, prefer an L2 workflow

## 3. Plan

- present the proposed approach before any high-cost action
- show defaults, assumptions, risks, and meaningful options
- reference the artifacts that will be produced next

## 4. Confirm

User confirmation is mandatory before:

- installing or upgrading packages
- overwriting files or directories
- running heavy PTQ or evaluation jobs
- generating or modifying execution scripts
- applying fixes that change user code or command lines

The confirm step must include:

- what will happen next
- what paths or environments are affected
- what defaults were chosen
- what the user can change before execution

## 5. Execute Or Summarize

- execute only after confirmation when execution was requested
- otherwise produce a runbook, commands, and next steps
- always summarize produced artifacts, key decisions, and known risks

## Recovery Rule

If a skill cannot continue, it must return:

- the blocking reason
- the artifact or precondition that is missing
- the smallest next action that can unblock the user
