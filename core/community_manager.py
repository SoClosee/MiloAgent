import json
import logging
import random
time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Default flair templates per subreddit
DEFAULT_FLAIRS = [
    {"text": "Discussion", "css_class": "discussion"},
    {"text": "Question", "css_class": "question"},
    {"text": "Tool / Resource", "css_class": "tool"},
    {"text": "Guide / Tutorial", "css_class": "guide"},
    {"text": "News", "css_class": "news"},
]

# Safety limits
MAX_MOD_ACTIONS_PER_CYCLE = 20
MAX_RULES_PER_SUB = 10
TAKEOVER_REQUEST_COOLDOWN_DAYS = 15  # Reddit's minimum between r/redditrequest posts

class CommunityManager:
    """Manages owned subreddit lifecycle: creation → setup → moderation → growth."""

    def __init__(self, db, llm, content_gen, hub_manager, intel):
        self.db = db
        self.llm = llm
        self.content_gen = content_gen
        self.hub_manager = hub_manager
        self.intel = intel
        self._ensure_tables()

    def _ensure_tables(self):
        """Create tracking tables if not exists."""
        try:
            with self.db._lock:
                self.db.conn.execute("BEGIN TRANSACTION")
                self.db.conn.executescript(
                    ""\"
                        CREATE TABLE IF NOT EXISTS community_setup_log (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            subreddit TEXT NOT NULL,
                            project TEXT NOT NULL,
                            step TEXT NOT NULL,
                            status TEXT DEFAULT 'pending',
                            completed_at TEXT,
                            details TEXT,
                            UNIQUE(subreddit, step)
                        );

                        CREATE TABLE IF NOT EXISTS subreddit_requests (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            subreddit TEXT NOT NULL,
                            project TEXT NOT NULL,
                            account TEXT NOT NULL,
                            request_post_url TEXT,
                            submitted_at TEXT DEFAULT (datetime('now')),
                            status TEXT DEFAULT 'pending',
                            checked_at TEXT,
                            reason TEXT,
                            takeover_score REAL,
                            metadata TEXT,
                            UNIQUE(subreddit, account)
                        );

                        CREATE INDEX IF NOT EXISTS idx_setup_sub
                            ON community_setup_log(subreddit);
                        CREATE INDEX IF NOT EXISTS idx_requests_status
                            ON subreddit_requests(status);
                    """
                )
                self.db.conn.execute("COMMIT")
        except Exception as e:
            logger.debug(f"Community tables init: {e}")

        # Extend subreddit_hubs table with new columns (safe ALTER)
        new_columns = [
            ("setup_complete", "INTEGER DEFAULT 0"),
            ("automod_configured", "INTEGER DEFAULT 0"),
            ("rules_count", "INTEGER DEFAULT 0"),
            ("flair_count", "INTEGER DEFAULT 0"),
            ("sticky_post_1", "TEXT"),
            ("sticky_post_2", "TEXT"),
            ("fullname", "TEXT"),
            ("ownership_type", "TEXT DEFAULT 'created'"),
            ("mod_queue_last_checked", "TEXT"),
        ]
        for col_name, col_type in new_columns:
            try:
                self.db.conn.execute(
                    f"ALTER TABLE subreddit_hubs ADD COLUMN {col_name} {col_type}"
                )
            except Exception:
                pass  # Column already exists

    # ── Subreddit Setup Pipeline ─────────────────────────────────────

    def get_setup_status(self, subreddit: str) -> Dict:
        """Check which setup steps have been completed for a subreddit."""
        steps = {}
        try:
            rows = self.db.conn.execute(
                "SELECT step, status FROM community_setup_log WHERE subreddit = ?",
                (subreddit,),
            ).fetchall()
            for row in rows:
                steps[row["step"]] = row["status"]
        except Exception:
            pass
        return steps

    def _mark_step(self, subreddit: str, project: str, step: str,
                   status: str = "completed", details: str = ""):
        """Mark a setup step as completed or failed."""
        try:
            self.db._execute_write(
                """
                    INSERT INTO community_setup_log (subreddit, project, step, status, completed_at, details)
                       VALUES (?, ?, ?, ?, datetime('now'), ?)
                       ON CONFLICT(subreddit, step) DO UPDATE SET
                       status=excluded.status, completed_at=excluded.completed_at, details=excluded.details
                """,
                (subreddit, project, step, status, details),
            )
        except Exception as e:
            logger.debug(f"Mark step error: {e}")

    def setup_new_subreddit(self, reddit_bot, subreddit: str, project: Dict) -> bool:
        """Full setup of a newly created subreddit.

        Runs the complete pipeline: settings → rules → flairs → automod → stickies.
        Skips already-completed steps (idempotent).
        """
        proj_name = project.get("project", {}).get("name", "unknown")
        status = self.get_setup_status(subreddit)
        logger.info(f"Setting up r/{subreddit} for {proj_name} (done: {list(status.keys())})")

        # Generate config via LLM
        config = self._generate_subreddit_config(subreddit, project)
        if not config:
            logger.error(f"Failed to generate config for r/{subreddit}")
            return False

        success_count = 0

        # Step 1: Update settings (sidebar, description)
        if status.get("settings") != "completed":
            if self._apply_settings(reddit_bot, subreddit, config, proj_name):
                success_count += 1
            time.sleep(random.uniform(3, 8))

        # Step 2: Create rules
        if status.get("rules") != "completed":
            if self._apply_rules(reddit_bot, subreddit, config, proj_name):
                success_count += 1
            time.sleep(random.uniform(3, 8))

        # Step 3: Create flairs
        if status.get("flairs") != "completed":
            if self._apply_flairs(reddit_bot, subreddit, config, proj_name):
                success_count += 1
            time.sleep(random.uniform(3, 8))

        # Step 4: Configure AutoModerator
        if status.get("automod") != "completed":
            if self._apply_automod(reddit_bot, subreddit, config, project, proj_name):
                success_count += 1
            time.sleep(random.uniform(3, 8))

        # Step 5: Create and pin welcome post
        if status.get("welcome_post") != "completed":
            if self._create_welcome_post(reddit_bot, subreddit, config, project, proj_name):
                success_count += 1
            time.sleep(random.uniform(5, 15))

        # Step 6: Create and pin rules post
        if status.get("rules_post") != "completed":
            if self._create_rules_post(reddit_bot, subreddit, config, project, proj_name):
                success_count += 1

        # Mark overall setup complete
        all_done = all(
            self.get_setup_status(subreddit).get(s) == "completed"
            for s in ("settings", "rules", "flairs", "automod", "welcome_post", "rules_post")
        )
        if all_done:
            try:
                self.db._execute_write(
                    "UPDATE subreddit_hubs SET setup_complete = 1 WHERE subreddit = ?",
                    (subreddit,),
                )
            except Exception:
                pass
            logger.info(f"r/{subreddit} setup complete!")

        return all_done

    def _generate_subreddit_config(self, subreddit: str, project: Dict) -> Optional[Dict]:
        """Use LLM to generate appropriate settings for a new subreddit."""
        proj_info = project.get("project", {})
        proj_name = proj_info.get("name", "")
        desc = proj_info.get("description", "")
        audiences = proj_info.get("target_audiences", [])
        reddit_cfg = project.get("reddit", {})
        owned = reddit_cfg.get("owned_subreddits", [])

        # Find matching config for this subreddit
        sub_config = {}
        for s in owned:
            if s.get("name", "").lower() == subreddit.lower():
                sub_config = s
                break

        niche = sub_config.get("niche", desc)
        title = sub_config.get("title", f"r/{subreddit}")

        prompt = f"""You are setting up a new Reddit community: r/{subreddit}
Title: {title}
Niche: {niche}
Related project: {proj_name} — {desc}
Target audience: {', '.join(audiences[:5])}

Generate a complete subreddit configuration. Output in this EXACT format:

SIDEBAR:
[Write 200-400 words of sidebar markdown. Include: welcome message, what the community is about, useful links, how to participate. Make it welcoming and professional.]

PUBLIC_DESCRIPTION:
[One sentence, max 200 chars, for the subreddit's public description.]

RULES:
1. [Rule name] | [Brief description of the rule]
2. [Rule name] | [Brief description]
3. [Rule name] | [Brief description]
4. [Rule name] | [Brief description]
5. [Rule name] | [Brief description]

WELCOME_TITLE:
[Title for the pinned welcome post]

WELCOME_BODY:
[2-3 paragraphs welcoming new members, explaining the community purpose, and encouraging participation.]

Guidelines:
- Be genuine and community-focused, not corporate
- Rules should protect quality without being overly strict
- Include a rule that allows product/tool sharing with context (this lets us share our links)
- Include a no-spam rule (but define spam as low-effort, not all links)
- Write in English
"""

        try:
            response = self.llm.generate(prompt, task="creative", max_tokens=2000)
            return self._parse_config_response(response)
        except Exception as e:
            logger.error(f"Config generation failed: {e}")
            return None

    def _parse_config_response(self, response: str) -> Dict:
        """Parse LLM config response into structured dict."""
        config = {
            "sidebar": "",
            "public_description": "",
            "rules": [],
            "welcome_title": "",
            "welcome_body": "",
        }

        current_section = None
        buffer = []

        for line in response.split("\n"):
            stripped = line.strip()

            if stripped.startswith("SIDEBAR:"):
                if current_section and buffer:
                    config[current_section] = "\n".join(buffer).strip()
                current_section = "sidebar"

... (truncated, 798 more lines)