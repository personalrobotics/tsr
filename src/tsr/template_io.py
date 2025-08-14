"""TSR Template I/O utilities for YAML file management."""

import os
import yaml
from pathlib import Path
from typing import List, Dict, Union, Optional
from .core.tsr_template import TSRTemplate
from .schema import EntityClass, TaskCategory, TaskType


class TemplateIO:
    """Utilities for reading and writing TSR template YAML files."""
    
    @staticmethod
    def save_template(template: TSRTemplate, filepath: Union[str, Path]) -> None:
        """Save a single TSR template to a YAML file.
        
        Args:
            template: The TSR template to save
            filepath: Path to the output YAML file
            
        Example:
            >>> template = TSRTemplate(...)
            >>> TemplateIO.save_template(template, "templates/mug_side_grasp.yaml")
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            yaml.dump(template.to_dict(), f, default_flow_style=False, sort_keys=False)
    
    @staticmethod
    def load_template(filepath: Union[str, Path]) -> TSRTemplate:
        """Load a single TSR template from a YAML file.
        
        Args:
            filepath: Path to the input YAML file
            
        Returns:
            The loaded TSR template
            
        Example:
            >>> template = TemplateIO.load_template("templates/mug_side_grasp.yaml")
        """
        filepath = Path(filepath)
        
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        
        return TSRTemplate.from_dict(data)
    
    @staticmethod
    def save_template_collection(
        templates: List[TSRTemplate], 
        filepath: Union[str, Path]
    ) -> None:
        """Save multiple TSR templates to a single YAML file.
        
        Args:
            templates: List of TSR templates to save
            filepath: Path to the output YAML file
            
        Example:
            >>> templates = [template1, template2, template3]
            >>> TemplateIO.save_template_collection(templates, "templates/grasps.yaml")
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = [template.to_dict() for template in templates]
        
        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    @staticmethod
    def load_template_collection(filepath: Union[str, Path]) -> List[TSRTemplate]:
        """Load multiple TSR templates from a single YAML file.
        
        Args:
            filepath: Path to the input YAML file
            
        Returns:
            List of loaded TSR templates
            
        Example:
            >>> templates = TemplateIO.load_template_collection("templates/grasps.yaml")
        """
        filepath = Path(filepath)
        
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        
        if not isinstance(data, list):
            raise ValueError(f"Expected list of templates in {filepath}")
        
        return [TSRTemplate.from_dict(template_data) for template_data in data]
    
    @staticmethod
    def load_templates_from_directory(
        directory: Union[str, Path],
        pattern: str = "*.yaml"
    ) -> List[TSRTemplate]:
        """Load all TSR templates from a directory.
        
        Args:
            directory: Directory containing template YAML files
            pattern: File pattern to match (default: "*.yaml")
            
        Returns:
            List of loaded TSR templates
            
        Example:
            >>> templates = TemplateIO.load_templates_from_directory("templates/grasps/")
        """
        directory = Path(directory)
        templates = []
        
        for filepath in directory.glob(pattern):
            try:
                # Try to load as single template first
                template = TemplateIO.load_template(filepath)
                templates.append(template)
            except Exception as e:
                # If that fails, try as collection
                try:
                    collection = TemplateIO.load_template_collection(filepath)
                    templates.extend(collection)
                except Exception as e2:
                    print(f"Warning: Could not load {filepath}: {e2}")
        
        return templates
    
    @staticmethod
    def load_templates_by_category(
        base_directory: Union[str, Path],
        categories: Optional[List[str]] = None
    ) -> Dict[str, List[TSRTemplate]]:
        """Load TSR templates organized by category.
        
        Args:
            base_directory: Base directory containing category subdirectories
            categories: List of categories to load (default: all)
            
        Returns:
            Dictionary mapping category names to lists of templates
            
        Example:
            >>> templates_by_category = TemplateIO.load_templates_by_category("templates/")
            >>> grasps = templates_by_category["grasps"]
            >>> places = templates_by_category["places"]
        """
        base_directory = Path(base_directory)
        templates_by_category = {}
        
        if categories is None:
            # Load all categories
            categories = [d.name for d in base_directory.iterdir() if d.is_dir()]
        
        for category in categories:
            category_dir = base_directory / category
            if category_dir.exists():
                templates = TemplateIO.load_templates_from_directory(category_dir)
                templates_by_category[category] = templates
        
        return templates_by_category
    
    @staticmethod
    def save_templates_by_category(
        templates_by_category: Dict[str, List[TSRTemplate]],
        base_directory: Union[str, Path]
    ) -> None:
        """Save TSR templates organized by category.
        
        Args:
            templates_by_category: Dictionary mapping category names to lists of templates
            base_directory: Base directory to save category subdirectories
            
        Example:
            >>> templates_by_category = {
            ...     "grasps": [grasp1, grasp2],
            ...     "places": [place1, place2]
            ... }
            >>> TemplateIO.save_templates_by_category(templates_by_category, "templates/")
        """
        base_directory = Path(base_directory)
        base_directory.mkdir(parents=True, exist_ok=True)
        
        for category, templates in templates_by_category.items():
            category_dir = base_directory / category
            category_dir.mkdir(exist_ok=True)
            
            for template in templates:
                # Generate filename from template properties
                filename = f"{template.subject_entity.value}_{template.reference_entity.value}_{template.task_category.value}_{template.variant}.yaml"
                filepath = category_dir / filename
                TemplateIO.save_template(template, filepath)
    
    @staticmethod
    def validate_template_file(filepath: Union[str, Path]) -> bool:
        """Validate that a YAML file contains a valid TSR template.
        
        Args:
            filepath: Path to the YAML file to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            TemplateIO.load_template(filepath)
            return True
        except Exception:
            return False
    
    @staticmethod
    def get_template_info(filepath: Union[str, Path]) -> Dict:
        """Get metadata about a TSR template without loading it completely.
        
        Args:
            filepath: Path to the YAML file
            
        Returns:
            Dictionary containing template metadata
        """
        filepath = Path(filepath)
        
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
        
        if isinstance(data, list):
            # Collection file
            return {
                'type': 'collection',
                'count': len(data),
                'templates': [
                    {
                        'name': t.get('name', ''),
                        'subject_entity': t.get('subject_entity', ''),
                        'reference_entity': t.get('reference_entity', ''),
                        'task_category': t.get('task_category', ''),
                        'variant': t.get('variant', '')
                    }
                    for t in data
                ]
            }
        else:
            # Single template file
            return {
                'type': 'single',
                'name': data.get('name', ''),
                'subject_entity': data.get('subject_entity', ''),
                'reference_entity': data.get('reference_entity', ''),
                'task_category': data.get('task_category', ''),
                'variant': data.get('variant', ''),
                'description': data.get('description', '')
            }


# Convenience functions for common operations
def save_template(template: TSRTemplate, filepath: Union[str, Path]) -> None:
    """Save a single TSR template to a YAML file."""
    TemplateIO.save_template(template, filepath)


def load_template(filepath: Union[str, Path]) -> TSRTemplate:
    """Load a single TSR template from a YAML file."""
    return TemplateIO.load_template(filepath)


def save_template_collection(templates: List[TSRTemplate], filepath: Union[str, Path]) -> None:
    """Save multiple TSR templates to a single YAML file."""
    TemplateIO.save_template_collection(templates, filepath)


def load_template_collection(filepath: Union[str, Path]) -> List[TSRTemplate]:
    """Load multiple TSR templates from a single YAML file."""
    return TemplateIO.load_template_collection(filepath)


def get_package_templates() -> Path:
    """Get the path to templates included in the package."""
    try:
        import tsr
        return Path(tsr.__file__).parent / "templates"
    except ImportError:
        # Fallback for development
        return Path(__file__).parent / "templates"


def list_available_templates() -> List[str]:
    """List all templates available in the package."""
    template_dir = get_package_templates()
    if not template_dir.exists():
        return []
    
    templates = []
    for yaml_file in template_dir.rglob("*.yaml"):
        templates.append(str(yaml_file.relative_to(template_dir)))
    return sorted(templates)


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
    return TemplateIO.load_templates_from_directory(category_dir)
