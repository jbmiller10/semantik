"""init_collections_schema

Revision ID: b6af1f8a14e8
Revises: 860fb2e922f2
Create Date: 2025-07-15 15:25:43.925650

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b6af1f8a14e8'
down_revision: Union[str, Sequence[str], None] = '860fb2e922f2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
