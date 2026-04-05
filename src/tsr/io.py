# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Siddhartha Srinivasa

"""TSR Template I/O utilities for YAML file management."""

import logging
from pathlib import Path
from typing import List, Union

import yaml

from .template import TSRTemplate

logger = logging.getLogger(__name__)


def save_template(template: TSRTemplate, filepath: Union[str, Path]) -> None:
    """Save a single TSR template to a YAML file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        yaml.dump(template.to_dict(), f, default_flow_style=False, sort_keys=False)


def load_template(filepath: Union[str, Path]) -> TSRTemplate:
    """Load a single TSR template from a YAML file."""
    with open(filepath, "r") as f:
        data = yaml.safe_load(f)
    return TSRTemplate.from_dict(data)


def save_template_collection(templates: List[TSRTemplate], filepath: Union[str, Path]) -> None:
    """Save multiple TSR templates to a single YAML file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    data = [template.to_dict() for template in templates]
    with open(filepath, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def load_template_collection(filepath: Union[str, Path]) -> List[TSRTemplate]:
    """Load multiple TSR templates from a single YAML file."""
    with open(filepath, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected list of templates in {filepath}")
    return [TSRTemplate.from_dict(d) for d in data]


def load_templates_from_directory(directory: Union[str, Path], pattern: str = "*.yaml") -> List[TSRTemplate]:
    """Load all TSR templates from a directory."""
    directory = Path(directory)
    templates = []
    for filepath in sorted(directory.glob(pattern)):
        try:
            templates.append(load_template(filepath))
        except (yaml.YAMLError, KeyError, ValueError, TypeError) as e:
            logger.warning("Could not load %s: %s", filepath, e)
    return templates


def get_package_templates() -> Path:
    """Get the path to templates included in the package."""
    return Path(__file__).parent / "templates"


def list_available_templates() -> List[str]:
    """List all templates available in the package."""
    template_dir = get_package_templates()
    if not template_dir.exists():
        return []
    return sorted(str(f.relative_to(template_dir)) for f in template_dir.rglob("*.yaml"))


def load_package_template(category: str, name: str) -> TSRTemplate:
    """Load a specific template from the package."""
    template_path = get_package_templates() / category / name
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")
    return load_template(template_path)


def load_package_templates_by_category(category: str) -> List[TSRTemplate]:
    """Load all templates from a specific category in the package."""
    category_dir = get_package_templates() / category
    if not category_dir.exists():
        return []
    return load_templates_from_directory(category_dir)


# Backwards compatibility
class TemplateIO:
    """Deprecated: use module-level functions directly."""

    save_template = staticmethod(save_template)
    load_template = staticmethod(load_template)
    save_template_collection = staticmethod(save_template_collection)
    load_template_collection = staticmethod(load_template_collection)
    load_templates_from_directory = staticmethod(load_templates_from_directory)
