Prompt: What would you change on reflection?

Good question. Let me think honestly about this rather than just defending what we've built.

**Things I'd change:**


**The Slack channel structure is overengineered.** Six channels for one person is noise. I'd start with two: `#job-search` for everything operational (briefings, materials review, approvals) and `#job-agents-ops` for system health. Split channels out later if the volume justifies it. You'll actually engage more with a single busy channel than six quiet ones.

**The heartbeat every 30 minutes is too frequent for approval checking.** It means the Coordinator is burning Anthropic tokens 48 times a day just to check if you've typed "approved" in Slack. A 2-hour heartbeat during business hours with a `/jobhunt check` slash command for immediate processing would be more cost-effective. Or better — configure a Slack event trigger so the Coordinator wakes _only_ when you actually post in the channel, rather than polling.

**The pipeline state machine is too granular for the database.** DISCOVERED → SHORTLISTED → MATERIALS_READY → AWAITING_APPROVAL is four states before you've even applied. Your existing system has a simpler model (New → Saved → Applied → Interviewing → Offered/Rejected) that works fine. I'd keep your existing statuses and track the sub-states (materials prep, approval pending) as agent-side context in the filesystem rather than database columns. Don't over-schema early.

**I underweighted the master CV challenge.** The entire system's output quality depends on how good `master-cv.md` is, and I treated it as "create a file, populate it." In reality, this needs significant upfront effort — a comprehensive, well-structured document with every project, metric, skill, and achievement you might want to reference. It's the foundational input to every tailored CV and cover letter. I'd add a dedicated "Phase 0" that's purely about building and validating the master CV, possibly using an Opus session to help structure it from your existing CV, LinkedIn profile, and project notes.

**The company research caching needs a TTL.** I specified writing research briefs to disk but didn't address staleness. A company brief written in week 2 might be cited by the Cover Letter agent in week 8, referencing outdated news. Each brief should have a "researched date" and the Cover Letter agent should re-research if the brief is older than 14 days.

**I didn't adequately address the "cold start" problem.** When you first deploy, there's no pipeline history, no response rate data, and no learned patterns for the Coordinator's weekly reviews to analyse. The first 2-3 weeks will produce fairly generic strategy memos. I should have specified that the Coordinator's early weekly reviews focus on establishing baselines rather than recommending changes, and that strategy adjustments kick in only after 15-20 applications have been sent.

**The security model assumes honest agents.** I locked down file permissions and network access, but an LLM agent with `exec` tool access can run arbitrary shell commands. If a prompt injection via a job description causes the agent to run something unexpected, the filesystem permissions won't help — the agent process owns those files. I should have recommended a sandboxing layer (running agents in a restricted shell, or Docker containers with limited mount points) rather than relying solely on file permissions. At minimum, the `exec` tool should have an allowlist of permitted commands rather than full shell access.

**What I wouldn't change:** the core architecture (OpenClaw as orchestrator, `exec + curl` to FastAPI, filesystem for documents, SQLite for pipeline state), the decision to skip n8n initially, the mandatory approval gates on all external communication, and the phased implementation approach. Those are the load-bearing decisions and they're sound.
