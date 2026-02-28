# Business (project) manager with hot-reload and CRUD operations.

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
            logger.info(f"Projects reloaded: {[p["project"]["name"] for p in projects]}")
        elif not old_names:
            logger.info(f"No projects found.")

        # Call reload callbacks
        for callback in self._on_reload_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error calling reload callback: {e}")

    def add_on_reload_callback(self, callback: Callable):
        """Register a callback to be called on project reload."""
        self._on_reload_callbacks.append(callback)

    def remove_on_reload_callback(self, callback: Callable):
        """Unregister a previously registered callback."""
        try:
            self._on_reload_callbacks.remove(callback)
        except ValueError:
            pass
