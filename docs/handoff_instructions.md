# VoxCPM2 Implementation Handoff Instructions

## Session Start Protocol

### 1. Read These Documents First

**Required reading order:**

1. `/a0/usr/projects/voxcpm_rs/AGENTS.md` — Root DOX framework + project rules
2. `/a0/usr/projects/voxcpm_rs/docs/voxcpm2_migration_plan.md` — Full 20-step migration plan
3. `/a0/usr/projects/voxcpm_rs/docs/implementation.md` — Current progress (if exists)
4. `/a0/usr/projects/voxcpm_rs/src/AGENTS.md` — Library module documentation
5. `/a0/usr/projects/voxcpm_rs/src/bin/AGENTS.md` — Binary entry point documentation

### 2. Check Current State

```bash
cd /a0/usr/projects/voxcpm_rs
git status
git log --oneline -5
```

### 3. Identify Next Step

- If `docs/implementation.md` exists, read it to find the last completed step
- The next step is the first one with `Status: In Progress` or no entry
- If all steps are complete, the migration is done

---

## Subagent Delegation Protocol

### When to Use Subagents

**Delegate to subagents for:**
- Individual implementation steps (Phase 0-8)
- Code changes with clear scope and success criteria
- Independent verification tasks

**Keep in main agent for:**
- Session coordination and progress tracking
- Cross-step dependencies and blocking issues
- Final validation and git commits
- Plan deviations and architectural decisions

### Subagent Prompt Template

When delegating a step to a subagent, use this structure:

```
## Role
Rust developer implementing VoxCPM2 migration step.

## Context
- Project: /a0/usr/projects/voxcpm_rs (VoxCPM Rust/Burn implementation)
- Branch: openai
- Current step: Step X.Y - [Step Name]
- Migration plan: /a0/usr/projects/voxcpm_rs/docs/voxcpm2_migration_plan.md

## Task
[Copy the exact step description from the migration plan]

## Constraints
- Follow DOX framework in AGENTS.md
- Read relevant AGENTS.md files before editing code
- Make minimal focused changes matching existing style
- Use `cargo fmt` before completing
- Do not edit tests/docs/lockfiles unless required

## Success Criteria
[Copy the exact success criteria from the migration plan]

## Verification Commands
[Copy the exact verification commands from the migration plan]

## Deliverables
1. Code changes that compile with `cargo build --release`
2. Verification output proving success criteria met
3. Summary of changes made
```

---

## Step Execution Workflow

### For Each Step

1. **Read the step** from `voxcpm2_migration_plan.md`
2. **Update implementation.md** with start timestamp
3. **Delegate to subagent** with the prompt template above
4. **Validate subagent work**:
   - Run `cargo build --release` to verify compilation
   - Run verification commands from the plan
   - Check that success criteria are met
5. **Git commit** changes:
   ```bash
   git add -A
   git commit -m "step X.Y: [brief description]"
   ```
6. **Update implementation.md** with completion details
7. **Move to next step** or checkpoint session

### Validation Checklist

Before marking a step complete:

- [ ] Code compiles without errors
- [ ] Verification commands produce expected output
- [ ] Success criteria explicitly met
- [ ] Changes committed to git
- [ ] implementation.md updated
- [ ] No unintended side effects

---

## Session Checkpoint Protocol

### Before Ending Session

1. Update `docs/implementation.md` with current status
2. Commit all changes:
   ```bash
   git add -A
   git commit -m "checkpoint: session end, last step X.Y"
   ```
3. Note next step to resume in implementation.md

### After Resuming Session

1. Read `docs/implementation.md` for current state
2. Check git status for uncommitted changes
3. Resume from the first incomplete step
4. Review any deviations noted in implementation.md

---

## Deviation Handling

If you must deviate from the plan:

1. **Document the deviation** in implementation.md:
   - What changed
   - Why it was necessary
   - Impact on future steps
2. **Update the migration plan** if the deviation affects subsequent steps
3. **Commit both files** together
4. **Note for future sessions** that the deviation exists

---

## Quick Reference

| Document | Location | Purpose |
|----------|----------|---------|
| Migration Plan | `docs/voxcpm2_migration_plan.md` | 20-step implementation plan |
| Progress Tracker | `docs/implementation.md` | Session checkpointing |
| Root DOX | `AGENTS.md` | Project rules + DOX framework |
| Library DOX | `src/AGENTS.md` | Module documentation |
| Binary DOX | `src/bin/AGENTS.md` | Binary documentation |
| Upstream 1.5 | `/tmp/voxcpm_1.5/` | Reference Python code |
| Upstream main | `/tmp/voxcpm_main/` | VoxCPM2 reference code |

---

## First Session Setup

For the very first implementation session:

1. Create `docs/implementation.md` with this template:

```markdown
# VoxCPM2 Implementation Progress

## Session Log

### Session 1
- **Date**: [current date]
- **Agent**: [agent info]
- **Steps Completed**: None yet
- **Next Step**: Step 0.1 - Install huggingface-hub Python Package

---

## Step Progress

### Step 0.1: Install huggingface-hub Python Package
- **Status**: Not Started
- **Started**: -
- **Completed**: -
- **Verification**: -
- **Success Criteria Met**: -
- **Git Commit**: -
- **Deviations**: -

[... repeat for all 20 steps ...]
```

2. Begin with **Phase 0: Environment & Weight Preparation**
3. Follow the subagent delegation protocol for each step

