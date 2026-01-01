"""Create a new plugin project scaffold."""

from __future__ import annotations

from pathlib import Path

import click
from jinja2 import Environment, PackageLoader

PLUGIN_TYPES = ["embedding", "chunking", "connector"]


def to_snake_case(value: str) -> str:
    return value.strip().replace("-", "_").replace(" ", "_").lower()


def to_pascal_case(value: str) -> str:
    words = value.replace("-", " ").replace("_", " ").split()
    return "".join(word.capitalize() for word in words)


def _template_paths(plugin_type: str, template_variant: str) -> list[str]:
    paths = [
        "common/README.md.j2",
        "common/.gitignore.j2",
        f"{plugin_type}/pyproject.toml.j2",
        f"{plugin_type}/plugin.py.j2",
        f"{plugin_type}/__init__.py.j2",
    ]
    if template_variant == "full":
        paths.append(f"{plugin_type}/tests/test_contract.py.j2")
    return paths


def _output_path(template_name: str, project_dir: Path, module_name: str) -> Path:
    if template_name.endswith("README.md.j2"):
        return project_dir / "README.md"
    if template_name.endswith(".gitignore.j2"):
        return project_dir / ".gitignore"
    if template_name.endswith("pyproject.toml.j2"):
        return project_dir / "pyproject.toml"
    if template_name.endswith("plugin.py.j2"):
        return project_dir / "src" / module_name / "plugin.py"
    if template_name.endswith("__init__.py.j2"):
        return project_dir / "src" / module_name / "__init__.py"
    if "tests/test_contract.py.j2" in template_name:
        return project_dir / "tests" / "test_contract.py"
    return project_dir / template_name.replace(".j2", "")


@click.command()
@click.argument("name")
@click.option(
    "--type",
    "plugin_type",
    "-t",
    type=click.Choice(PLUGIN_TYPES),
    required=True,
    help="Type of plugin to create",
)
@click.option(
    "--output",
    "-o",
    "output_dir",
    type=click.Path(),
    default=".",
    help="Output directory",
)
@click.option(
    "--template",
    type=click.Choice(["minimal", "full"]),
    default="full",
    help="Template variant",
)
def new(name: str, plugin_type: str, output_dir: str, template: str) -> None:
    """Create a new Semantik plugin project.

    Example:
        semantik-plugin new my-embedder --type embedding
    """
    env = Environment(loader=PackageLoader("semantik_cli", "templates"), autoescape=False)

    project_dir = Path(output_dir) / name
    project_dir.mkdir(parents=True, exist_ok=True)

    module_name = to_snake_case(name)
    class_name = to_pascal_case(name)

    context = {
        "name": name,
        "module_name": module_name,
        "class_name": class_name,
        "plugin_type": plugin_type,
    }

    for template_name in _template_paths(plugin_type, template):
        output_path = _output_path(template_name, project_dir, module_name)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        content = env.get_template(template_name).render(**context)
        output_path.write_text(content)

    click.echo(f"Created plugin project at {project_dir}")
    click.echo("Next steps:")
    click.echo(f"  cd {project_dir}")
    click.echo("  pip install -e .")
    click.echo("  semantik-plugin validate .")
