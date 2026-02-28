"""""Business (project) manager with hot-reload and CRUD operations."""

import os
import time
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Callable

import yaml

logger = logging.getLogger(__name__)

class BusinessManager:
    """Manages project YAML files with hot-reload capability.

    Features:
    - Load all projects from projects/ directory
    - Watch for file changes (add/edit/delete) via mtime polling
    - CRUD operations for projects
    - Thread-safe access to projects list
    - Callbacks for reload notification
    """

    def __init__(self, projects_dir: str = "projects/"):
        self.projects_dir = Path(projects_dir)
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        self._projects: List[Dict] = []
        self._file_mtimes: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._watcher_thread: Optional[threading.Thread] = None
        self._watching = False
        self._on_reload_callbacks: List[Callable] = []

        # Initial load
        self.reload()

    @property
    def projects(self) -> List[Dict]:
        """Thread-safe access to current projects list."""
        with self._lock:
            return list(self._projects)

    def reload(self):
        """Reload all projects from disk."""
        projects = []
        new_mtimes = {}

        for f in self.projects_dir.glob("*.yaml"):
            try:
                new_mtimes[str(f)] = f.stat().st_mtime
                with open(f) as fh:
                    data = yaml.safe_load(fh) or {}
                if data and data.get("project", {}).get("enabled", True):
                    projects.append(data)
            except Exception as e:
                logger.error(f"Error loading project file {f}: {e}")

        projects.sort(
            key=lambda p: p.get("project", {}).get("weight", 1.0),
            reverse=True,
        )

        with self._lock:
            old_names = {p["project"]["name"] for p in self._projects}
            new_names = {p["project"]["name"] for p in projects}
            self._projects = projects
            self._file_mtimes = new_mtimes

        # Log changes
        added = new_names - old_names
        removed = old_names - new_names
        if added:
            logger.info(f"Projects added: {added}")
        if removed:
            logger.info(f"Projects removed: {removed}")
        if not added and not removed and old_names:
            logger.info(f"Projects reloaded: {[p['project']['name'] for p in projects]}")
        elif not old_names:
            logger.info(f"Loaded {len(projects)} projects: {[p['project']['name'] for p in projects]}")

        # Notify callbacks
        for cb in self._on_reload_callbacks:
            try:
                cb(projects)
            except Exception as e:
                logger.error(f"Reload callback error: {e}")

    # ── File Watcher ──────────────────────────────────────────────────

    def start_watching(self, projects_dir: str = "projects/"):
        """Start watching for file changes in the projects directory."""
        self.projects_dir = Path(projects_dir)
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        self._watcher_thread = threading.Thread(target=self._watch_files)
        self._watcher_thread.start()

    def _watch_files(self):
        """Thread target to watch for file changes and reload projects."""
        while True:
            time.sleep(5)  # Check every 5 seconds
            new_mtimes = {}
            for f in self.projects_dir.glob("*.yaml"):
                new_mtimes[str(f)] = f.stat().st_mtime
            if new_mtimes != self._file_mtimes:
                self.reload()

    def stop_watching(self):
        """Stop the file watcher thread."""
        if self._watcher_thread and self._watcher_thread.is_alive():
            self._watcher_thread.join()
            self._watcher_thread = None

    # ── CRUD Operations ─────────────────────────────────────────────

    def create_project(self, project_data: Dict):
        """Create a new project."""
        with open(f"{self.projects_dir}/{project_data['name']}.yaml", "w") as fh:
            yaml.dump(project_data, fh)
        self.reload()

    def update_project(self, project_name: str, project_data: Dict):
        """Update an existing project."""
        with open(f"{self.projects_dir}/{project_name}.yaml", "w") as fh:
            yaml.dump(project_data, fh)
        self.reload()

    def delete_project(self, project_name: str):
        """Delete a project."""
        os.remove(f"{self.projects_dir}/{project_name}.yaml")
        self.reload()
""