# Perplexity API Integration for Job Hunt Project

**Phase:** 5B (Research Depth Enhancement)  
**Status:** Ready for Implementation  
**Owner:** Research Depth Specialist  
**Last Updated:** 2026-02-25

---

## Executive Summary

This document outlines the integration of Perplexity API (sonar-pro model) into the Job Hunt workflow to provide deep company intelligence, competitive analysis, and strategic insights that go beyond what Brave Search can offer.

**Key Benefits:**
- 📊 Synthesized company intelligence from multiple sources with citations
- 🎯 Recent developments focus (M&A, funding, strategic shifts)
- 💡 Contextual insights for cover letters and interview prep
- 💰 Cost-effective (~$5-6/month for expected usage)

**Estimated ROI:** 10x improvement in research quality for ~$0.05 per company deep-dive

---

## API Overview

### Available Models

| Model | Use Case | Cost | Recommended |
|-------|----------|------|-------------|
| **sonar-pro** | Company research, competitive intelligence | $3/$15 per M tokens + ~$0.01 request | ✅ **YES** |
| sonar | Simple queries, basic search | $1/$1 per M tokens | ❌ Too basic |
| sonar-reasoning-pro | Multi-step reasoning tasks | $2/$8 per M tokens | ❌ Overkill |
| sonar-deep-research | Academic research reports | $2-8 per M + query fees | ❌ Too expensive |

### Why sonar-pro?

- **Best balance:** Quality synthesis vs. cost
- **Built-in citations:** Automatic source attribution
- **200k context window:** Can handle complex queries
- **Recent info:** Web-grounded, current as of query time
- **Query cost:** ~$0.02-0.05 typical company research query

---

## API Setup

### 1. Get API Key

1. Sign up at [Perplexity API Platform](https://www.perplexity.ai/api-platform)
2. Navigate to API Keys tab
3. Generate new key (starts with `pplx-`)
4. Store in 1Password: `Perplexity API Key`

### 2. Install SDK

```bash
pip install perplexityai
```

### 3. Set Environment Variable

Add to `~/.zshrc`:
```bash
export PERPLEXITY_API_KEY="pplx-xxxxxxxxxxxxx"
```

### 4. Test Installation

```bash
cd ~/.openclaw/skills/perplexity/scripts
./test_perplexity.py
```

Expected output:
- ✅ Response from API with company intelligence
- 📊 Token usage stats
- 🔗 Citations extracted
- 💰 Cost breakdown (~$0.02-0.05)

---

## Architecture Design

### Decision Flow: When to Use Perplexity vs. Brave

```
User Query
    |
    v
Is it simple factual lookup? ──YES──> Brave Search (free)
    |                                  - "What is Company X?"
   NO                                  - "Where is HQ?"
    |                                  - "Job posting link"
    v
Is it company intelligence? ──YES──> Perplexity API (sonar-pro)
    |                                  - Company strategy
   NO                                  - Recent developments
    |                                  - Competitive positioning
    v                                  - Culture/values
Brave Search (broad topics)
```

### Integration Points

#### Market Intel Agent
**Trigger:** Daily scan of top target companies  
**Query Pattern:** "What are [Company]'s strategic priorities and recent initiatives in [relevant area] as of 2026?"  
**Frequency:** 3 queries/day (top targets)  
**Cost:** ~$0.12/day = $3.60/month  
**Output:** `~/projects/claude_jobhunt/research/{company_name}.md`

#### Interview Prep Agent
**Trigger:** Interview scheduled with company  
**Query Pattern:** "Recent news about [Company]: M&A, funding, leadership changes, strategic shifts (last 6 months)"  
**Frequency:** On-demand (~5/week = 20/month)  
**Cost:** ~$1.00/month  
**Output:** Briefing doc with talking points

#### Cover Letter Agent
**Trigger:** Drafting cover letter for specific role  
**Query Pattern:** "What are [Company]'s core values, culture, and recent achievements in [job area]?"  
**Frequency:** 1-2 per application (10-20/month)  
**Cost:** ~$0.50/month  
**Output:** Key hooks to reference in letter

### Caching Strategy

**Problem:** Don't want to re-query same company repeatedly  
**Solution:** File-based cache with TTL

```python
# Cache structure: ~/.openclaw/workspace/perplexity-cache/
{company_slug}/
  ├── metadata.json         # Last query time, cost, citations
  ├── strategic-overview.md # General company intel
  ├── recent-news.md        # 6-month news scan
  └── culture-values.md     # Values, culture, initiatives
```

**Cache TTL:**
- Strategic overview: 30 days
- Recent news: 7 days
- Culture/values: 30 days

**Cache hit ratio target:** 60% (saves ~$2-3/month)

---

## Query Patterns

### 1. Company Strategic Overview

**Use Case:** New target company, need comprehensive understanding

```python
query = f"""What are {company_name}'s strategic priorities and key initiatives in 2026? 
Focus on:
- Core business strategy and recent pivots
- Technology investments (especially AI, data, automation)
- Market positioning and competitive advantages
- Recent growth areas or new service lines

Provide specific examples and initiatives."""

parameters = {
    "model": "sonar-pro",
    "max_tokens": 1500,
    "temperature": 0.2,
    "search_recency_filter": "month",
    "search_domain_filter": [f"{company_domain}"],  # Prioritize official sources
}
```

**Expected cost:** $0.03-0.05  
**Cache for:** 30 days

---

### 2. Recent Developments Scan

**Use Case:** Interview prep, need latest news

```python
query = f"""Recent news and developments about {company_name} in the last 6 months:
- Mergers, acquisitions, or partnerships
- Funding rounds or financial results
- Leadership changes or organizational restructuring
- Major product launches or strategic announcements
- Industry recognition or awards

Prioritize business-critical developments."""

parameters = {
    "model": "sonar-pro",
    "max_tokens": 1200,
    "temperature": 0.2,
    "search_recency_filter": "month",
}
```

**Expected cost:** $0.02-0.04  
**Cache for:** 7 days

---

### 3. Culture & Values Deep-Dive

**Use Case:** Cover letter hooks, culture fit assessment

```python
query = f"""What are {company_name}'s core values, company culture, and employee experience?
Include:
- Official values and mission statement
- Real employee experiences (from Glassdoor, LinkedIn, etc.)
- Diversity & inclusion initiatives
- Work-life balance and benefits philosophy
- Career development and growth opportunities

Be specific about {relevant_department} if possible."""

parameters = {
    "model": "sonar-pro",
    "max_tokens": 1500,
    "temperature": 0.3,
    "search_domain_filter": [f"{company_domain}", "glassdoor.com", "linkedin.com"],
}
```

**Expected cost:** $0.03-0.05  
**Cache for:** 30 days

---

### 4. Competitive Positioning

**Use Case:** Market intel, understanding differentiation

```python
query = f"""How does {company_name} differentiate itself from competitors like {competitor_1} and {competitor_2} in {market_segment}?
Include:
- Unique value propositions
- Technology or methodology advantages
- Market share and positioning
- Client segments and industries served
- Pricing or delivery model differences"""

parameters = {
    "model": "sonar-pro",
    "max_tokens": 1500,
    "temperature": 0.2,
}
```

**Expected cost:** $0.04-0.06  
**Cache for:** 30 days

---

## Response Parsing & Citation Handling

### Citation Extraction

Perplexity responses include inline citations like `[1]`, `[2]` in the text, with URLs in the `citations` array.

```python
def parse_perplexity_response(response):
    """Extract structured data from Perplexity response."""
    content = response.choices[0].message.content
    citations = response.citations if hasattr(response, 'citations') else []
    usage = response.usage
    
    # Calculate cost
    input_cost = (usage.prompt_tokens / 1_000_000) * 3
    output_cost = (usage.completion_tokens / 1_000_000) * 15
    request_fee = 0.010  # Medium search context estimate
    total_cost = input_cost + output_cost + request_fee
    
    # Extract citation numbers from text
    import re
    citation_refs = re.findall(r'\[(\d+)\]', content)
    
    return {
        'content': content,
        'citations': citations,
        'citation_refs': sorted(set(int(x) for x in citation_refs)),
        'usage': {
            'prompt_tokens': usage.prompt_tokens,
            'completion_tokens': usage.completion_tokens,
            'total_tokens': usage.total_tokens
        },
        'cost': total_cost,
        'cost_breakdown': {
            'input': input_cost,
            'output': output_cost,
            'request': request_fee
        }
    }
```

### Saving Research Output

**Directory structure:**
```
~/projects/claude_jobhunt/research/
├── {company_slug}/
│   ├── strategic-overview.md
│   ├── recent-news.md
│   ├── culture-values.md
│   └── competitive-positioning.md
└── .cache/
    └── {company_slug}-metadata.json
```

**Metadata file format:**
```json
{
  "company_name": "KPMG",
  "company_slug": "kpmg",
  "last_updated": "2026-02-25T19:30:00Z",
  "queries": [
    {
      "type": "strategic-overview",
      "timestamp": "2026-02-25T19:30:00Z",
      "cost": 0.0423,
      "tokens": {
        "prompt": 127,
        "completion": 1342,
        "total": 1469
      },
      "citations": [
        "https://kpmg.com/strategy-2026",
        "https://example.com/kpmg-ai-initiative"
      ]
    }
  ],
  "total_cost": 0.0423,
  "cache_valid_until": {
    "strategic-overview": "2026-03-27T19:30:00Z",
    "recent-news": "2026-03-03T19:30:00Z"
  }
}
```

---

## Cost Controls

### Daily Query Limits

**Limit:** Maximum 5 Perplexity queries per day across all agents

**Implementation:**
```python
def check_daily_limit():
    """Check if daily query limit reached."""
    log_file = "~/.openclaw/workspace/perplexity-costs.json"
    
    try:
        with open(log_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        return True  # No log yet, allow query
    
    today = datetime.now().date()
    today_queries = [
        q for q in data.get('queries', [])
        if datetime.fromisoformat(q['timestamp']).date() == today
    ]
    
    if len(today_queries) >= 5:
        print("⚠️  Daily Perplexity query limit (5) reached. Falling back to Brave Search.")
        return False
    
    return True
```

### Cost Tracking

**Log file:** `~/.openclaw/workspace/perplexity-costs.json`

```json
{
  "total_spent": 2.47,
  "queries": [
    {
      "timestamp": "2026-02-25T19:30:00Z",
      "agent": "market-intel",
      "company": "KPMG",
      "query_type": "strategic-overview",
      "cost": 0.0423,
      "tokens": 1469
    }
  ],
  "monthly_summary": {
    "2026-02": {
      "spent": 2.47,
      "query_count": 58,
      "avg_cost_per_query": 0.0426
    }
  }
}
```

### Budget Alerts

```python
def check_monthly_budget():
    """Alert if approaching monthly budget limit."""
    MONTHLY_BUDGET = 10.00  # Conservative limit
    
    log_file = "~/.openclaw/workspace/perplexity-costs.json"
    
    try:
        with open(log_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        return True
    
    current_month = datetime.now().strftime("%Y-%m")
    month_spent = data.get('monthly_summary', {}).get(current_month, {}).get('spent', 0)
    
    if month_spent >= MONTHLY_BUDGET * 0.8:  # 80% threshold
        print(f"⚠️  Perplexity budget alert: ${month_spent:.2f} / ${MONTHLY_BUDGET:.2f} used this month")
        if month_spent >= MONTHLY_BUDGET:
            print("🛑 Monthly budget exceeded. Disabling Perplexity until next month.")
            return False
    
    return True
```

---

## Agent-Specific Use Cases

### Market Intel Agent

**Goal:** Daily monitoring of top 3-5 target companies

**Workflow:**
1. Check cache for each target company
2. If strategic overview >30 days old → query Perplexity (strategic-overview pattern)
3. If recent news >7 days old → query Perplexity (recent-developments pattern)
4. Save outputs to research directory
5. Generate daily briefing with new insights

**Queries per day:** 2-3 (with 60% cache hit rate)  
**Monthly cost:** ~$3.00

---

### Interview Prep Agent

**Goal:** Deep brief for upcoming interview

**Workflow:**
1. **Triggered:** User schedules interview with company
2. **Check cache:** Load existing research if <7 days old
3. **Deep queries:**
   - Strategic overview (if not cached)
   - Recent developments (always fresh)
   - Culture & values (if not cached)
4. **Generate briefing:**
   - Executive summary of company
   - Recent news & strategic moves
   - Questions to ask interviewer
   - Culture fit assessment
5. **Prepare talking points:** Link personal experience to company initiatives

**Queries per interview:** 1-3 (depending on cache)  
**Average cost per interview:** $0.05-0.10  
**Monthly interviews:** ~10  
**Monthly cost:** ~$1.00

---

### Cover Letter Agent

**Goal:** Company-specific hooks and value alignment

**Workflow:**
1. **Triggered:** User requests cover letter for specific role
2. **Query Perplexity:** Culture & values pattern + strategic overview (if not cached)
3. **Extract hooks:**
   - Company values that align with user's experience
   - Recent initiatives user can reference
   - Team/department-specific context
4. **Inject into cover letter:**
   - "I was particularly impressed by [specific initiative from research]..."
   - "Your focus on [value] aligns with my experience in [relevant project]..."

**Queries per application:** 1-2  
**Average cost per cover letter:** $0.03-0.05  
**Applications per month:** ~10  
**Monthly cost:** ~$0.50

---

## Testing Checklist

### Pre-Integration Testing

- [x] Install perplexityai SDK
- [x] Set PERPLEXITY_API_KEY environment variable
- [ ] Run `test_perplexity.py` and verify:
  - [ ] Response received successfully
  - [ ] Content is relevant and synthesized
  - [ ] Citations are extracted correctly
  - [ ] Cost is within expected range ($0.02-0.05)
  - [ ] Token usage is reasonable (<2000 tokens)

### Integration Testing

- [ ] **Market Intel Agent:**
  - [ ] Trigger research on new company
  - [ ] Verify output saved to research directory
  - [ ] Confirm citations included
  - [ ] Check cache TTL set correctly
  - [ ] Test cache hit on second query
  
- [ ] **Interview Prep Agent:**
  - [ ] Trigger interview prep briefing
  - [ ] Verify recent news query works
  - [ ] Confirm talking points generated
  - [ ] Check cost tracking logged
  
- [ ] **Cover Letter Agent:**
  - [ ] Request cover letter with company research
  - [ ] Verify company-specific hooks included
  - [ ] Confirm values alignment extracted
  - [ ] Check total cost per application

### Cost Control Testing

- [ ] Daily limit enforcement:
  - [ ] Make 5 queries in one day
  - [ ] Verify 6th query blocked
  - [ ] Confirm fallback to Brave Search
  
- [ ] Monthly budget alert:
  - [ ] Simulate spending 80% of budget
  - [ ] Verify warning message shown
  - [ ] Test hard stop at 100% budget

### Error Handling Testing

- [ ] Rate limit (429) handling:
  - [ ] Force rate limit (burst 20+ queries)
  - [ ] Verify exponential backoff works
  - [ ] Confirm fallback to Brave Search
  
- [ ] Invalid API key:
  - [ ] Test with wrong key
  - [ ] Verify error message clear
  - [ ] Confirm graceful degradation
  
- [ ] Network timeout:
  - [ ] Simulate network failure
  - [ ] Verify retry logic works
  - [ ] Check timeout after 30s

---

## Monthly Cost Estimates

### Conservative Estimate (Expected Usage)

| Agent | Queries/Day | Queries/Month | Cost/Query | Monthly Cost |
|-------|-------------|---------------|------------|--------------|
| Market Intel | 2-3 | 60-90 | $0.04 | $2.40-3.60 |
| Interview Prep | 0.5 | 10-15 | $0.05 | $0.50-0.75 |
| Cover Letter | 0.3 | 10 | $0.04 | $0.40 |
| **Total** | **2.8-3.8** | **80-115** | **$0.04** | **$3.30-4.75** |

**With 60% cache hit rate:** ~$2.00-3.00/month

### Aggressive Estimate (Heavy Usage)

| Agent | Queries/Day | Queries/Month | Cost/Query | Monthly Cost |
|-------|-------------|---------------|------------|--------------|
| Market Intel | 5 | 150 | $0.05 | $7.50 |
| Interview Prep | 1 | 30 | $0.05 | $1.50 |
| Cover Letter | 0.5 | 15 | $0.04 | $0.60 |
| **Total** | **6.5** | **195** | **$0.05** | **$9.60** |

**With 40% cache hit rate:** ~$5.50-6.50/month

**Budget recommendation:** Set $10/month hard limit for safety margin

---

## Fallback Strategy

### When Perplexity Unavailable

1. **Rate limited (429):** Wait with exponential backoff, then fallback to Brave
2. **Daily limit reached:** Fallback to Brave Search immediately
3. **Monthly budget exceeded:** Disable Perplexity until next month, use Brave
4. **API error:** Retry once, then fallback to Brave
5. **Network timeout:** Fallback to Brave after 30s

### Graceful Degradation

- **Without Perplexity:** Agents still function using Brave Search
- **Quality impact:** Less synthesis, no citations, more manual research needed
- **User notification:** "Using Brave Search (Perplexity limit reached)"

---

## Implementation Roadmap

### Phase 1: Core Infrastructure (Week 1)
- [x] Research Perplexity API and create SKILL.md
- [ ] Test API with prototype script
- [ ] Set up research directory structure
- [ ] Implement cost tracking system
- [ ] Create cache management utilities

### Phase 2: Agent Integration (Week 2)
- [ ] Update Market Intel agent prompt
- [ ] Update Interview Prep agent prompt
- [ ] Update Cover Letter agent prompt
- [ ] Add Perplexity query functions to each agent
- [ ] Implement cache checks before querying

### Phase 3: Cost Controls (Week 3)
- [ ] Implement daily query limits
- [ ] Add monthly budget tracking
- [ ] Set up alert thresholds
- [ ] Test fallback logic
- [ ] Document cost monitoring dashboard

### Phase 4: Testing & Refinement (Week 4)
- [ ] End-to-end testing with all agents
- [ ] Measure cache hit ratio
- [ ] Optimize query patterns for cost
- [ ] Tune model parameters (temperature, max_tokens)
- [ ] Document lessons learned

---

## Success Metrics

### Quality Metrics
- **Citation quality:** >90% of responses have relevant, authoritative citations
- **Synthesis quality:** Responses combine 3+ sources (vs. Brave's single-snippet approach)
- **Recency:** 80% of company intelligence is <30 days old

### Cost Metrics
- **Average cost per query:** <$0.05
- **Monthly total spend:** <$6.00 (with cache)
- **Cache hit ratio:** >60%
- **Cost per application:** <$0.15 (all research combined)

### Usage Metrics
- **Daily queries:** 2-4 on average (within 5 limit)
- **Market Intel adoption:** Used for 80% of target companies
- **Interview Prep adoption:** Used for 90% of interviews
- **Cover Letter adoption:** Used for 60% of applications

---

## Monitoring & Maintenance

### Daily
- Check `perplexity-costs.json` for anomalies
- Review query count vs. daily limit
- Spot-check research output quality

### Weekly
- Review cache hit ratio (target: >60%)
- Analyze cost trends
- Adjust query patterns if needed
- Clean up old cache files (>90 days)

### Monthly
- Generate cost report and compare to budget
- Review agent adoption rates
- Tune query patterns based on usage
- Update model selection if Perplexity releases new models

---

## Troubleshooting

### "Rate limit exceeded"
**Cause:** Too many queries in short time  
**Fix:** Exponential backoff, then fallback to Brave  
**Prevention:** Implement request throttling (1 query per 10s)

### "Citations not found"
**Cause:** Response format changed or parsing error  
**Fix:** Check `response.citations` attribute vs. parsing inline `[1]` refs  
**Prevention:** Test with latest SDK regularly

### "Cost higher than expected"
**Cause:** Long responses, complex queries, or high search context  
**Fix:** Reduce `max_tokens`, add more specific queries, use cache  
**Prevention:** Set `max_tokens: 1500` default, monitor token usage

### "Cache not invalidating"
**Cause:** TTL logic error or manual cache updates  
**Fix:** Check metadata.json timestamps, force refresh if needed  
**Prevention:** Automated cache cleanup script (weekly cron)

---

## Related Documentation

- [Perplexity API Official Docs](https://docs.perplexity.ai)
- [SKILL.md](~/.openclaw/skills/perplexity/SKILL.md) - Detailed usage guide
- [test_perplexity.py](~/.openclaw/skills/perplexity/scripts/test_perplexity.py) - Test script
- [Job Hunt Agents Overview](~/projects/claude_jobhunt/docs/AGENTS.md)
- [Market Intel Agent](~/projects/claude_jobhunt/agents/market-intel/README.md)
- [Interview Prep Agent](~/projects/claude_jobhunt/agents/interview-prep/README.md)

---

**Questions or issues?** Contact Research Depth Specialist (this subagent) or review logs at `~/.openclaw/workspace/perplexity-costs.json`
