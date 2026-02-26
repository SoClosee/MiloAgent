<p align="center">
  <img src="https://img.shields.io/badge/MiloAgent-v3.0-blueviolet?style=for-the-badge&logo=robot&logoColor=white" alt="MiloAgent v3.0"/>
</p>

<h1 align="center">MiloAgent</h1>

<p align="center">
  <strong>AI-Powered Social Media Growth Engine</strong><br>
  <em>Autonomous bot that grows your product's presence on Reddit, Twitter/X & Telegram</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+"/>
  <img src="https://img.shields.io/badge/cost-$0%2Fmonth-brightgreen?style=flat-square" alt="Zero Cost"/>
  <img src="https://img.shields.io/badge/LLM-Groq%20|%20Gemini%20|%20Ollama-orange?style=flat-square" alt="LLM Providers"/>
  <img src="https://img.shields.io/badge/platforms-Reddit%20|%20Twitter%20|%20Telegram-blue?style=flat-square" alt="Platforms"/>
  <img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="MIT License"/>
  <img src="https://img.shields.io/badge/docker-ready-2496ED?style=flat-square&logo=docker&logoColor=white" alt="Docker Ready"/>
</p>

---

## What is MiloAgent?

MiloAgent is a fully autonomous growth bot that promotes your product/project across social media by joining real conversations naturally. It scans Reddit, Twitter/X, and Telegram for relevant discussions, generates human-like comments using LLM, and posts them — all while learning from the results to get better over time.

**No API costs.** Runs entirely on free-tier LLMs (Groq + Gemini).

### Key Features

| Feature | Description |
|---------|-------------|
| **Multi-Platform** | Reddit (web + API), Twitter/X, Telegram groups |
| **Smart Content** | LLM-generated comments adapted to context, subreddit culture, and post type |
| **Self-Learning** | Learns which subreddits, tones, and post types perform best |
| **A/B Testing** | Automatically tests tone, content type, length, and promo ratio |
| **Multi-Account** | Rotate between accounts with independent cooldowns |
| **Anti-Ban** | Rate limiting, human-like delays, shadowban detection, circuit breaker |
| **3 Dashboards** | Web UI, Terminal TUI, Telegram bot — monitor from anywhere |
| **Zero Cost** | Groq + Gemini free tiers = $0/month |
| **Docker Ready** | One-command deployment with docker-compose |
| **Expert Personas** | 10+ domain-expert personas that adapt to each community |
| **Community Management** | Create and manage your own subreddits |
| **Hot-Reload** | Edit project configs — changes apply instantly, no restart |

---

## How It Works

```
Every 8 minutes:
  SCAN → Find relevant posts across Reddit/Twitter/Telegram
    ↓
  SCORE → Rank opportunities by relevance, freshness, engagement
    ↓
  GENERATE → Create a natural comment via LLM (80% organic / 20% promo)
    ↓
  VALIDATE → Check for spam patterns, links, duplicates
    ↓
  POST → Publish the comment with human-like timing
    ↓
  LEARN → Track performance, adjust strategy, improve prompts
```

### Content Types

The bot generates two types of content in a configurable ratio:

**Organic (80%)** — Genuine, helpful responses with no product mention:
> "I had the same problem — what worked for me was splitting long videos into 60s segments and posting the highlights. Took some trial and error but now my workflow is pretty smooth. What platform are you targeting first?"

**Promotional (20%)** — Natural product mentions in context:
> "I've been using [YourProduct] to automate that process — the AI picks out the key moments and exports them in the right format. Saved me hours compared to manual editing. Have you tried any automated tools yet?"

---

## Quick Start

### Prerequisites

- Python 3.10+
- A free [Groq API key](https://console.groq.com) or [Gemini API key](https://aistudio.google.com/apikey)
- At least one Reddit account

### Installation

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/MiloAgent.git
cd MiloAgent

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install Playwright (for browser-based login)
playwright install chromium
```

### Configuration

1. **Add your LLM API key** — edit `config/llm.yaml`:
```yaml
providers:
  groq:
    api_key: "gsk_your_key_here"    # Free at https://console.groq.com
```

2. **Add your Reddit account** — edit `config/reddit_accounts.yaml`:
```yaml
accounts:
  - username: "your_reddit_username"
    password: "your_password"
    enabled: true
    assigned_projects: ["my_project"]
```

3. **Create your project** — copy and edit `projects/example_project.yaml`:
```bash
cp projects/example_project.yaml projects/my_project.yaml
# Edit with your product details, target subreddits, keywords, etc.
```

4. **Login to Reddit** (captures session cookies):
```bash
python3 miloagent.py login reddit
```

5. **Verify everything works**:
```bash
python3 miloagent.py setup     # Check configuration
python3 miloagent.py test all  # Test all connections
```

### Run

```bash
# Start the bot (foreground)
python3 miloagent.py run

# Start as daemon (background)
python3 miloagent.py run --daemon

# Stop the daemon
python3 miloagent.py stop
```

---

## Docker Deployment

```bash
# Copy and edit environment variables
cp .env.example .env
nano .env

# Build and start
docker compose up -d

# View logs
docker compose logs -f
```

### Server Deployment (VPS)

```bash
# First-time setup (Nginx + SSL + .env)
./deploy.sh --setup

# Deploy
./deploy.sh --up

# Update
./deploy.sh --update

# Status
./deploy.sh --status
```

---

## CLI Reference

| Command | Description |
|---------|-------------|
| `setup` | Verify configuration |
| `run` | Start the bot (`--daemon` for background) |
| `stop` | Stop the daemon |
| `dashboard` | Launch Terminal UI (TUI) |
| `scan reddit` | Scan Reddit for opportunities |
| `post reddit -p my_project` | Post a comment |
| `post reddit -p my_project --dry-run` | Generate without posting |
| `engage all` | Organic engagement (upvote, subscribe, follow) |
| `login reddit` | Login via browser |
| `login twitter` | Login via browser |
| `paste-cookies reddit` | Paste cookies manually |
| `test all` | Test all connections |
| `status` | Quick status overview |
| `stats -h 24` | Detailed stats (last 24h) |
| `accounts` | Account health status |
| `learn` | Run learning cycle |
| `insights` | View learned patterns |
| `business list` | List projects |
| `business add` | Add a new project |
| `business show <name>` | Show project config |
| `hub list` | List owned subreddits |
| `hub suggest -p <project>` | Get subreddit name suggestions |
| `hub create <name>` | Create a new subreddit |
| `system health` | System health check |
| `system cleanup` | Clean database & temp files |

---

## Telegram Dashboard

Control and monitor MiloAgent from Telegram:

| Command | Description |
|---------|-------------|
| `/status` | Bot state, RAM, recent actions |
| `/stats` | Last 24h statistics |
| `/report` | Full daily report |
| `/last 10` | Last 10 actions with content |
| `/insights` | Learned patterns & A/B results |
| `/intel` | Subreddit opportunity analysis |
| `/accounts` | Account health |
| `/scan` | Trigger a scan now |
| `/post` | Post now |
| `/pause` / `/resume` | Pause/resume the bot |
| `/performance` | Performance score & suggestions |

---

## Terminal Dashboard (TUI)

```bash
python3 miloagent.py dashboard
```

| Key | Action |
|-----|--------|
| `TAB` / `1-4` | Switch views (Main / Accounts / Conversations / Opportunities) |
| `s` | Scan | `a` | Act | `l` | Learn |
| `p` | Pause/Resume | `q` | Quit |
| `:` | Command mode (vim-style) |

---

## Architecture

```
miloagent.py (CLI)
    │
    ▼
Orchestrator (scheduler + job runner)
    │
    ├── ResourceMonitor     → CPU/RAM/Disk monitoring, auto-pause
    ├── BusinessManager     → Load projects, hot-reload
    ├── LearningEngine      → Self-improvement, A/B testing
    ├── ResearchEngine      → Trend analysis, news tracking
    │
    ├── Reddit (Web/API)    → Scan, comment, post, DM, verify
    ├── Twitter/X           → Scan, reply, tweet, DM
    ├── Telegram            → Group engagement, auto-discovery
    │
    ├── Safety Layer
    │   ├── RateLimiter     → Per-account, per-subreddit limits
    │   ├── BanDetector     → Shadowban detection
    │   ├── ContentDedup    → Prevent duplicate posts
    │   └── AccountManager  → Rotation, cooldowns, health
    │
    └── Dashboards
        ├── Web (FastAPI)   → Browser dashboard
        ├── TUI (Rich)      → Terminal dashboard
        └── Telegram Bot    → Mobile monitoring
```

---

## Project Structure

```
MiloAgent/
├── miloagent.py              # CLI entry point
├── requirements.txt          # Python dependencies
├── Dockerfile                # Container image
├── docker-compose.yml        # Container orchestration
├── deploy.sh                 # Server deployment script
├── miloagent.service         # Systemd service file
│
├── config/                   # Configuration (edit these!)
│   ├── settings.yaml         # Bot behavior & limits
│   ├── llm.yaml              # LLM provider API keys
│   ├── reddit_accounts.yaml  # Reddit credentials
│   ├── twitter_accounts.yaml # Twitter credentials
│   ├── telegram.yaml         # Telegram bot token
│   ├── telegram_user_accounts.yaml
│   └── expert_personas.yaml  # AI persona definitions
│
├── core/                     # Brain
│   ├── orchestrator.py       # Main loop & scheduler
│   ├── content_gen.py        # LLM content generation
│   ├── content_validator.py  # Pre-post validation
│   ├── learning_engine.py    # Self-improvement & A/B testing
│   ├── strategy.py           # Opportunity scoring
│   ├── research_engine.py    # Trend & news tracking
│   ├── business_manager.py   # Project management
│   ├── database.py           # SQLite data layer
│   ├── resource_monitor.py   # System resource monitoring
│   ├── cookie_manager.py     # Browser login & cookies
│   ├── subreddit_intel.py    # Subreddit analysis
│   ├── subreddit_hub.py      # Owned subreddit management
│   ├── community_manager.py  # Community moderation
│   ├── content_curator.py    # Content curation
│   ├── relationship_engine.py# Relationship building
│   └── ab_testing.py         # A/B experiment framework
│
├── platforms/                # Social media integrations
│   ├── base_platform.py      # Abstract interface
│   ├── reddit_web.py         # Reddit (cookie-based)
│   ├── reddit_bot.py         # Reddit (PRAW API)
│   ├── twitter_bot.py        # Twitter/X (Twikit)
│   └── telegram_group_bot.py # Telegram groups (Telethon)
│
├── safety/                   # Anti-ban protection
│   ├── rate_limiter.py       # Action limits & timing
│   ├── ban_detector.py       # Shadowban detection
│   ├── content_dedup.py      # Duplicate prevention
│   └── account_manager.py    # Account rotation & health
│
├── dashboard/                # Monitoring interfaces
│   ├── web.py                # FastAPI web dashboard
│   ├── tui.py                # Rich terminal UI
│   └── telegram_bot.py       # Telegram bot dashboard
│
├── prompts/                  # LLM prompt templates
│   ├── reddit_comment.txt
│   ├── reddit_post.txt
│   ├── twitter_reply.txt
│   ├── twitter_tweet.txt
│   ├── telegram_reply.txt
│   └── ... (18 templates)
│
├── projects/                 # Your products/projects (YAML)
│   └── example_project.yaml  # Template — copy and customize
│
├── data/                     # Runtime data (gitignored)
│   ├── miloagent.db          # SQLite database
│   ├── cookies/              # Session cookies
│   └── sessions/             # Telegram sessions
│
└── logs/                     # Log files (gitignored)
```

---

## Self-Learning System

MiloAgent continuously improves through 4 learning mechanisms:

### 1. Subreddit & Keyword Weighting
Tracks engagement per subreddit and keyword, automatically prioritizing high-performing targets.

### 2. Reply Sentiment Analysis
Analyzes responses to bot comments (positive: "thanks", "helpful" / negative: "spam", "bot") to adjust tone per subreddit. Zero LLM calls — pure keyword analysis.

### 3. A/B Testing
Automatically tests 4 variables (max 2 experiments simultaneously):
- **Tone** — which communication style gets the best engagement
- **Post type** — which content format performs best
- **Length** — short vs long responses
- **Promo ratio** — optimal promotional content percentage

### 4. Prompt Evolution
Evolves LLM prompt templates by analyzing top-performing posts. Auto-reverts if performance drops >30%.

---

## Anti-Ban Safety

| Protection | Details |
|-----------|---------|
| Rate Limiting | Per-account, per-subreddit, per-hour limits |
| Human Timing | Random delays, jitter on all actions |
| Shadowban Detection | Periodic profile & comment visibility checks |
| Circuit Breaker | 5 consecutive failures → pause requests |
| User-Agent Rotation | Pool of 5 browser user-agents |
| Resource Monitoring | Auto-pause if RAM > 90% or CPU > 80% |
| Content Validation | Checks for spam patterns before posting |
| Weekend Mode | 50% activity reduction on weekends |

---

## Zero-Cost Stack

| Component | Provider | Free Tier |
|-----------|----------|-----------|
| LLM (Primary) | Groq | 6,000 req/day |
| LLM (Fallback) | Gemini | 1,500 req/day |
| LLM (Local) | Ollama | Unlimited |
| Reddit | Web scraping | No API needed |
| Twitter | Twikit | Cookie-based |
| Database | SQLite | Local |
| Dashboard | FastAPI | Self-hosted |

**Estimated daily usage:** ~90 LLM calls/day (well within free tier)

---

## Automated Jobs

When running in `run` mode, these jobs execute automatically:

| Job | Frequency | Description |
|-----|-----------|-------------|
| Scan | 8 min | Find relevant posts across platforms |
| Act | 1-2 min | Post on the best opportunity |
| Engage | 2h | Organic engagement (upvotes, follows) |
| Verify | 1h | Check if comments were removed |
| Seed Content | 6h | Create original posts in target subs |
| Tweet Cycle | 45 min | Twitter engagement loop |
| Learn | 6h | Analyze performance, adjust weights |
| Auto-Improve | 12h | Self-optimize rate limits & prompts |
| Health Check | 30 min | Verify account health |
| Research | 4h | Track trends and news |
| Curate | 3h | Find and share relevant content |
| Build Relations | 3h | DM outreach & relationship building |
| Community | 2h | Maintain subreddit presence |
| Subreddit Intel | 8h | Deep subreddit analysis |
| Hub Animation | 6h | Animate owned subreddits |
| Daily Report | 24h | Telegram summary |

---

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>Built with Python, LLMs & patience</strong><br>
  <em>If you find this useful, give it a star!</em>
</p>
