---
aliases: [Job Hunt—Phase 4+ Planning]
linter-yaml-title-alias: Job Hunt—Phase 4+ Planning
date created: 2026-02-15 18:52:03 pm
date modified: 2026-03-18 20:17:29 pm
---

# Job Hunt—Phase 4+ Planning

**Created:** 2026-02-15

**Status:** Planning—not yet in development

---

## Strategic Direction

Nick's vision is evolving beyond "find jobs on Adzuna" into a **focused, intelligent job search operating system** with these key themes:

1. **Multi-source intelligence**—not just job boards, but LinkedIn recommendations, inbox signals, network referrals
2. **Deep research capability**—Perplexity for advanced analytics, not just Brave for basic search
3. **Signal over noise**—prioritisation, filtering, learning what resonates. Nick should see 3-5 high-quality opportunities, not 50 mediocre ones
4. **Autonomous daily operations**—the system runs itself, Nick reviews and decides
5. **Learning loop**—the system gets smarter about what Nick wants based on his approve/reject/skip patterns

---

## Phase 4: Interview Prep & Networking Agents

*Original scope from implementation guide*

### Interview Prep Agent

- Auto-triggers when a job moves to "interviewing" status
- Generates company research brief, likely interview questions, STAR-format answer suggestions
- Pulls from company news, Glassdoor, annual reports
- Posts prep pack to Slack

### Networking Agent + Contact Management UI

**Agent role:** Drafts outreach messages (LinkedIn, email) for target companies. All external communication requires Nick's explicit approval. Tracks response rates, optimises messaging.

**Contact management (Nick's priority):** A proper UI for managing networking contacts—not just a flat list but a prioritised, trackable pipeline.

**What Nick wants:**

- List of potential contacts with prioritisation (warm intro > cold outreach > speculative)
- Tracking: last contacted, response status, next action, notes
- Tagging: by company, by role type, by source (LinkedIn, referral, event, etc.)
- Ability to reorder/reprioritise manually
- Relationship warmth indicator (cold / warm / hot / active conversation)

**Backend (already exists):**

- `networking_contacts` table in SQLite with CRUD endpoints at `/api/agents/networking-contacts`
- Fields: name, company, role, relationship status, notes, timestamps
- **Needs extending:** priority field, tags, last_contacted, next_action, source, warmth score

**UI options:**

1. **Slack-based**—Coordinator posts contact list, Nick reacts/replies to prioritise. Simple but limited.
2. **Mission Control module**—Add a Networking plugin to the existing dashboard. Full table view, drag-to-reorder, filters, click-to-expand. Proper UI.
3. **Obsidian**—Dataview table sourced from markdown files. Nick already uses Obsidian. Low-tech but flexible.
4. **Standalone web app**—Overkill unless the other options fail.

**Recommendation:** Mission Control module (option 2) for the full experience, with Slack summaries for quick daily updates. The dashboard already has the FastAPI + Vue stack—adding a contacts table view is straightforward.

**Data flow:**

```
Networking Agent finds potential contact
  → Creates entry in DB via API
  → Posts to Slack: "Found [name] at [company] — [reason]. Priority: [X]"
  → Nick reviews in Slack (quick) or Mission Control (detailed)
  → Nick approves outreach / adjusts priority / adds notes
  → Agent drafts message → Nick approves → Agent tracks response
  → Learning: response rates feed back into prioritisation model
```

**Schema extension needed:**

```sql
ALTER TABLE networking_contacts ADD COLUMN priority INTEGER DEFAULT 3;  -- 1=urgent, 5=backburner
ALTER TABLE networking_contacts ADD COLUMN tags TEXT;  -- JSON array
ALTER TABLE networking_contacts ADD COLUMN last_contacted TIMESTAMP;
ALTER TABLE networking_contacts ADD COLUMN next_action TEXT;
ALTER TABLE networking_contacts ADD COLUMN next_action_date DATE;
ALTER TABLE networking_contacts ADD COLUMN source TEXT;  -- linkedin, referral, event, cold
ALTER TABLE networking_contacts ADD COLUMN warmth TEXT DEFAULT 'cold';  -- cold/warm/hot/active
ALTER TABLE networking_contacts ADD COLUMN response_status TEXT;  -- pending/replied/ghosted/meeting_set
```

---

## Phase 5+: Extended Capabilities

### 5A: LinkedIn Integration

**Problem:** LinkedIn sends well-tailored job recommendations to Nick's inbox. These are often higher quality than Adzuna because LinkedIn knows his profile, network, and preferences. Currently they're just sitting in email.

**Approach options (to investigate):**

1. **Email parsing**—Forward LinkedIn notification emails to a monitored inbox (school@jeevesbot.io pattern). Parse job titles, companies, links from the email HTML.
2. **LinkedIn API**—Limited for job search (LinkedIn restricts this). May not be viable.
3. **Browser automation**—Log into LinkedIn, scrape recommendations. Fragile, ToS-grey, but effective.
4. **Manual forwarding**—Nick forwards interesting LinkedIn jobs to Jeeves. Lowest tech, highest friction.

**Recommendation:** Start with email parsing (option 1). LinkedIn notifications have predictable HTML structure. Set up a forwarding rule from Nick's email → dedicated inbox → parsing agent. Low risk, no LinkedIn ToS issues.

**Key questions for Nick:**

- Which email receives LinkedIn notifications? (nicksolly@gmail.com?)
- Happy to set up a forwarding rule?
- Or would he prefer to manually forward interesting ones?

### 5B: Perplexity Integration (Advanced Research)

**Problem:** Brave Search is good for quick lookups but limited for deep research. The Market Intel and Interview Prep agents need richer context—company strategy, recent M&A, leadership changes, culture signals.

**What Perplexity adds:**

- Cited, synthesised answers (not just links)
- Better at "what is [company]'s AI strategy?" type queries
- Can summarise earnings calls, annual reports, news patterns
- Useful for competitive intelligence (who else is hiring for similar roles?)

**Integration plan:**

- Nick has an API key (to be shared)
- Add as a research tool alongside Brave Search
- Market Intel: use for company deep-dives on shortlisted leads
- Interview Prep: use for company research briefs
- Cover Letter: use for finding specific company hooks
- Coordinator: use for market trend analysis

**Architecture:** Add a `PERPLEXITY.md` skill file to relevant agent workspaces. Simple API call wrapper. Use Perplexity for depth, Brave for breadth.

### 5C: Daily Operations & Coordination

**Problem:** The system needs to run autonomously without overwhelming Nick. Currently no cron jobs are configured. Need to design the daily rhythm carefully.

**Proposed daily cycle:**

| Time | Agent | Action | Channel |
|------|-------|--------|---------|
| 06:00 | Market Intel | Overnight scan (Adzuna + LinkedIn inbox) |—|
| 07:00 | Coordinator | Morning briefing (top 3-5 opportunities, pipeline status, actions needed) | #daily |
| 12:00 | Market Intel | Midday scan |—|
| 12:15 | Coordinator | New leads alert (only if high-quality matches found) | #briefing |
| 18:00 | Market Intel | Evening scan |—|
| 18:15 | Coordinator | End-of-day summary (what happened, what needs Nick's attention) | #daily |
| 21:00 | Coordinator | Weekly summary (Fridays only) | #briefing |

**Key design principles:**

- **Coordinator is the single point of contact**—Nick talks to the Coordinator, not individual agents
- **Threshold-based alerts**—only surface jobs above a match threshold (configurable, starts high)
- **Batch over interrupt**—collect opportunities and present in batches, not one-by-one
- **Escalation hierarchy:** Routine → #daily digest. High match → #briefing alert. Urgent (deadline/referral) → Telegram DM
- **Quiet hours**—no Slack posts before 07:00 or after 21:00

### 5D: Learning Algorithm & Prioritisation

**Problem:** Match scoring is currently basic (profile keywords vs job description). Nick's approve/reject/skip patterns contain signal about what he actually wants.

**Learning signals to capture:**

- Jobs Nick saves → positive signal (role type, company tier, salary range, keywords)
- Jobs Nick skips → weak negative (maybe not relevant, maybe just busy)
- Jobs Nick explicitly rejects → strong negative (with reason if given)
- Jobs Nick applies to → strongest positive signal
- Time spent reviewing → engagement proxy
- Edit feedback on CVs/cover letters → what matters to Nick

**Implementation approach:**

1. **Short term:** Heuristic rules in Coordinator—boost companies/roles similar to approved ones, suppress similar to rejected ones. Store in a `preferences.json` that evolves.
2. **Medium term:** Weighted scoring model—each signal adjusts weights on role type, company, salary, skills, location, seniority.
3. **Long term:** Fine-tuned embeddings or classifier trained on Nick's feedback data.

**Start simple.** A `preferences.json` that the Coordinator reads and updates based on Nick's Slack responses would get us 80% of the value.

### 5E: Multi-Model Consensus Review

**Problem:** Current scoring is single-model, single-pass. One model's blind spots become the system's blind spots. At Director+ level, the difference between a great fit and a time-waster is nuanced—culture, trajectory, hidden requirements, political dynamics. We need deeper evaluation before surfacing anything to Nick.

**What Nick wants:**

- Multiple models reviewing the same opportunity independently
- Structured pros/cons analysis, not just a score
- Consensus (or flagged disagreement) before it reaches him
- Techniques beyond keyword matching—reasoning about career fit, growth trajectory, risk

**Available models:**

- **Anthropic** (Claude Opus, Sonnet)—via OpenClaw, primary
- **OpenRouter**—access to GPT-4o, Gemini, Llama, Mistral, DeepSeek, Cohere, etc.
- **Ollama**—local models for offline/cheap evaluation (already set up as backup)

**Proposed Architecture: The Review Panel**

```
Job enters pipeline (match score > threshold)
        │
        ▼
┌─────────────────────────────────────────┐
│         STAGE 1: Quick Screen           │
│  Single model (Sonnet — fast, cheap)    │
│  Binary: plausible fit? yes/no          │
│  Kills obvious mismatches early         │
└──────────────┬──────────────────────────┘
               │ (passes only)
               ▼
┌─────────────────────────────────────────┐
│      STAGE 2: Independent Reviews       │
│                                         │
│  Reviewer A: Claude Opus                │
│    → Strategic fit analysis             │
│    → Career trajectory assessment       │
│    → "Would this advance Nick's goals?" │
│                                         │
│  Reviewer B: GPT-4o (via OpenRouter)    │
│    → Skills gap analysis                │
│    → Salary/market assessment           │
│    → Red flag detection                 │
│                                         │
│  Reviewer C: Gemini (via OpenRouter)    │
│    → Company culture/reputation         │
│    → Role longevity/stability           │
│    → Hidden requirements in JD          │
│                                         │
│  (Reviews run in parallel, independent) │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│       STAGE 3: Consensus & Scoring      │
│  Sonnet synthesises all three reviews   │
│                                         │
│  Output:                                │
│  • Composite score (weighted average)   │
│  • Confidence level (agreement/dissent) │
│  • Structured pros/cons                 │
│  • Career fit narrative                 │
│  • Red flags (any reviewer can veto)    │
│  • Recommendation: PURSUE / CONSIDER /  │
│    SKIP / FLAG FOR NICK                 │
│                                         │
│  Disagreement = FLAG FOR NICK           │
│  (interesting roles split opinions)     │
└──────────────┬──────────────────────────┘
               │
               ▼
        Coordinator presents to Nick
        (only PURSUE + FLAG results)
```

**Evaluation dimensions (each reviewer scores 1-10):**

| Dimension | What it measures |
|-----------|-----------------|
| Role fit | Does the JD match Nick's experience and target? |
| Growth trajectory | Step up, lateral, or step down? Career momentum? |
| Company quality | Reputation, stability, culture, growth stage |
| Comp alignment | Salary vs target, total comp, equity potential |
| Skills match | Current skills coverage vs requirements |
| Skills growth | Would this role develop skills Nick wants? |
| Location/logistics | Commute, hybrid policy, travel requirements |
| Strategic value | Even if not perfect, does it open doors? |
| Red flags | Unrealistic requirements, high turnover signals, vague JD |
| Application effort | ROI—is the tailoring effort worth the probability of success? |

**Additional techniques to consider:**

1. **Devil's advocate prompting**—One model explicitly argues AGAINST applying. Forces the system to defend the recommendation.
2. **Calibration jobs**—Feed in roles Nick has already rejected/accepted. Use as reference points ("this is more like the KPMG role you liked than the Capita role you skipped").
3. **Blind review**—Strip company name from JD for one reviewer. Tests whether the role itself is interesting vs. brand appeal.
4. **Comparative ranking**—Don't just score in isolation. "Of the 5 leads this week, rank them." Relative comparison is often more useful than absolute scores.
5. **Ollama for pre-screening**—Run the cheapest local model first to kill obviously wrong jobs before spending API credits on the panel.

**Cost management:**

- Stage 1 (Sonnet quick screen): ~$0.01 per job
- Stage 2 (3 independent reviews): ~$0.15-0.30 per job
- Stage 3 (synthesis): ~$0.02 per job
- **Total: ~$0.20-0.35 per reviewed job**
- If 50 jobs/day pass Stage 1, and 10 reach Stage 2: ~$2-3.50/day
- Ollama pre-screen could cut Stage 2 volume by 50%: ~$1-2/day

**OpenRouter config needed:**

- API key (Nick to provide)
- Model selection: GPT-4o, Gemini 1.5 Pro, possibly DeepSeek for a third perspective
- Rate limits and fallback chain

**Implementation:** New `ReviewPanel` skill/module. Could live in the Coordinator or as a standalone evaluation service. Coordinator dispatches to panel before presenting to Nick.

---

### 5E-b: Focus Mode

**Problem:** 50 mediocre applications lose to 5 excellent ones. The system should help Nick focus, not just find.

**Features:**

- **Weekly target:** Coordinator sets a goal (e.g. "3 high-quality applications this week")
- **Pipeline limits:** Max 10 active applications at any time. Force prioritisation.
- **Quality gates:** CV Tailor and Cover Letter agents refuse to produce materials for jobs below a match threshold
- **Cool-down:** After applying somewhere, system waits before suggesting similar roles at the same company
- **Strategic recommendations:** "You've applied to 3 Big 4 roles this week. Consider diversifying—here's a FinTech option."

---

## Sequencing

| Priority | Item | Blocked by | Effort |
|----------|------|------------|--------|
| 1 | Cron jobs for daily operations | Nothing—do now | 2h |
| 2 | Perplexity integration | API key from Nick | 3h |
| 3 | LinkedIn email parsing | Email forwarding setup | 4h |
| 4 | Learning algorithm (preferences.json) | Some approve/reject data | 4h |
| 5 | Interview Prep agent | Nothing | 4h |
| 6 | Networking agent | Nothing | 4h |
| 7 | Focus mode | Learning algorithm | 3h |
| 8 | LinkedIn browser automation | Only if email parsing insufficient | 8h+ |

---

## 5F: Multi-Source Job Intelligence

**Problem:** Adzuna is one aggregator. At Director+ level in consulting/financial services, the best roles are often not on job boards at all. We need a broader funnel.

### Source Landscape (to research)

| Source | Type | Likely Quality for Nick | API/Scrape? | Notes |
|--------|------|------------------------|-------------|-------|
| **Adzuna** | Aggregator | Medium—broad but noisy | ✅ API (active) | Current source. Good volume, variable relevance |
| **LinkedIn Jobs** | Platform | High—tailored recs | Email parsing (planned) | Best signal-to-noise for senior roles |
| **Indeed** | Aggregator | Medium | API (limited) | Massive volume, lots of noise at senior level |
| **Glassdoor** | Platform + reviews | Medium | Scrape only | Useful for company intel more than job discovery |
| **Reed** | UK job board | Medium | API available | Strong UK market, especially finance/consulting |
| **Totaljobs** | UK job board | Medium | API available | Similar to Reed |
| **CWJobs / eFinancialCareers** | Specialist | High for FS/tech | Scrape | Niche boards for finance + technology |
| **Executive search firms** | Recruiters | Very High | Manual/email | Heidrick, Spencer Stuart, Odgers—Director+ is their bread and butter |
| **Company career pages** | Direct | High | Scrape | Target companies (KPMG, Deloitte, PwC, banks) directly |
| **Wellfound (AngelList)** | Startup | Medium-High for tech | API | Good for FinTech/AI startups |
| **Hired** | Tech platform | Medium-High | Manual | Companies apply to you. Good for senior tech. |
| **The Guardian Jobs** | Public sector | Low-Medium | Scrape | Occasionally has interesting GDS/public AI roles |
| **WorkInStartups** | UK startups | Medium | Scrape | UK-focused startup job board |
| **Networking/referrals** | Personal | Highest | Manual/email | The actual way most Director+ roles get filled |

### Research Questions

1. **Which sources have APIs?**—Reduces scraping fragility
2. **Which specialise in Director+ / £100k+?**—Executive-level boards we might be missing
3. **Which have the best UK consulting/FS coverage?**—Our specific niche
4. **Cost?**—Some APIs are paid (Indeed charges, LinkedIn is expensive)
5. **Deduplication**—Same role appears on multiple boards. Need content-hash dedup across sources.
6. **Executive search firms**—Can we monitor their public listings? Or is outreach the only way?

### Strategic Observation

At Nick's target level, the sourcing priority is roughly:

1. **Network & referrals** (highest hit rate, lowest visibility to system)
2. **Executive search / headhunters** (high quality, need relationship)
3. **LinkedIn** (good signal, already planned)
4. **Specialist boards** (eFinancialCareers, CWJobs)
5. **Company career pages** (targeted, high quality)
6. **Aggregators** (Adzuna, Indeed—volume play, diminishing returns at senior level)

The system should work its way UP this list, not just add more aggregators at the bottom. The Networking agent becomes critical here—it's not just about finding posted roles, it's about surfacing the ones that never get posted.

### Action: Research Sprint

When ready, spawn a research agent to:

- Survey available APIs and their terms/costs
- Identify the 3-4 highest-value additional sources for Nick's profile
- Prototype integration for the best one
- Document findings in Obsidian

---

## Open Questions for Nick

1. **LinkedIn emails**—which inbox? Happy with forwarding rule approach?
2. **Perplexity API key**—share when ready
3. **Daily rhythm**—does the proposed schedule work? Too much? Too little?
4. **Match threshold**—what score should trigger an alert vs. sit in the DB quietly?
5. **Pipeline limits**—comfortable with a max active applications cap?
6. **Slack channel structure**—keep current channels or create a dedicated #job-search?
7. **Telegram escalation**—want truly urgent opportunities (referral, perfect match) to ping Telegram?
