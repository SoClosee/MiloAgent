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
            logger.info(f"Projects reloaded: {[p["project"]["name"] for p in projects]}")

        # Reload environment variables from .env file
        self._load_env_vars()

    def _load_env_vars(self):
        """Reload environment variables from .env file."""
        env_file = Path(".env")
        if env_file.exists():
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        os.environ[key] = value

    # ── CRUD Operations ───────────────────────────────────────

    def create_project(self, project_data: Dict) -> bool:
        """Create a new project file."""
        try:
            with self._lock:
                project_file = self.projects_dir / f"{project_data["name"]}.yaml"
                with open(project_file, 'w') as fh:
                    yaml.dump(project_data, fh)
                self.reload()
                return True
        except Exception as e:
            logger.error(f"Error creating project file: {e}")
            return False

    def update_project(self, project_name: str, updated_data: Dict) -> bool:
        """Update an existing project file."""
        try:
            with self._lock:
                project_file = self.projects_dir / f"{project_name}.yaml"
                if project_file.exists():
                    with open(project_file, 'r') as fh:
                        current_data = yaml.safe_load(fh)
                    updated_data.update(current_data)
                    with open(project_file, 'w') as fh:
                        yaml.dump(updated_data, fh)
                    self.reload()
                    return True
                else:
                    logger.error(f"Project file not found: {project_name}")
                    return False
        except Exception as e:
            logger.error(f"Error updating project file: {e}")
            return False

    def delete_project(self, project_name: str) -> bool:
        """Delete a project file."""
        try:
            with self._lock:
                project_file = self.projects_dir / f"{project_name}.yaml"
                if project_file.exists():
                    project_file.unlink()
                    self.reload()
                    return True
                else:
                    logger.error(f"Project file not found: {project_name}")
                    return False
        except Exception as e:
            logger.error(f"Error deleting project file: {e}")
            return False

    # ── Callbacks ───────────────────────────────────────────────

    def add_on_reload_callback(self, callback: Callable):
        """Add a callback to be called on reload."""
        with self._lock:
            self._on_reload_callbacks.append(callback)

    def remove_on_reload_callback(self, callback: Callable):
        """Remove a callback from being called on reload."""
        with self._lock:
            self._on_reload_callbacks.remove(callback)

    # ── Modulo Operations ─────────────────────────────────────

    def _mod_operation(self, operation: str, data: Dict) -> bool:
        """Perform a modulo operation on project data."""
        try:
            with self._lock:
                if operation == "create":
                    return self.create_project(data)
                elif operation == "update":
                    return self.update_project(data["name"], data)
                elif operation == "delete":
                    return self.delete_project(data["name"])
                else:
                    logger.error(f"Invalid modulo operation: {operation}")
                    return False
        except Exception as e:
            logger.error(f"Error performing modulo operation: {e}")
            return False

    # ── Reload on Modulo Operations ────────────────────────────

    def reload_on_mod_operation(self, operation: str, data: Dict) -> bool:
        """Reload project data after performing a modulo operation."""
        try:
            with self._lock:
                if self._mod_operation(operation, data):
                    self.reload()
                    return True
                else:
                    return False
        except Exception as e:
            logger.error(f"Error reloading on modulo operation: {e}")
            return False

    # ── Reload on Modulo Operations with Callbacks ───────────────

    def reload_on_mod_operation_with_callbacks(self, operation: str, data: Dict) -> bool:
        """Reload project data after performing a modulo operation and call callbacks."""
        try:
            with self._lock:
                if self.reload_on_mod_operation(operation, data):
                    for callback in self._on_reload_callbacks:
                        callback()
                    return True
                else:
                    return False
        except Exception as e:
            logger.error(f"Error reloading on modulo operation with callbacks: {e}")
            return False