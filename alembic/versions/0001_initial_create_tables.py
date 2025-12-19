"""initial create users and file_assignments

Revision ID: 0001_initial
Revises: 
Create Date: 2025-02-16 00:00:00
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    # --- users table ---
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("full_name", sa.String(), nullable=False),
        sa.Column("college_id", sa.String(), nullable=False),
        sa.Column("college_name", sa.String(), nullable=False),
        sa.Column("mobile_number", sa.String(), nullable=False),
        sa.Column("email", sa.String(), nullable=False, unique=True, index=True),
        sa.Column("gender", sa.String(), nullable=True),
        sa.Column("age", sa.Integer(), nullable=True),
        sa.Column("dialect", sa.String(), nullable=True),
        sa.Column("is_admin", sa.Boolean(), nullable=False, server_default=sa.text('false')),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
        sa.Column("session_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("session_end", sa.DateTime(timezone=True), nullable=True),
    )

    # --- file_assignments table ---
    op.create_table(
        "file_assignments",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id")),
        sa.Column("base", sa.String(), nullable=False, index=True),
        sa.Column("audio_key", sa.String(), nullable=False),
        sa.Column("text_key", sa.String(), nullable=False),
        sa.Column("json_key", sa.String(), nullable=True),
        sa.Column("assigned_at", sa.DateTime(timezone=True), nullable=True, server_default=sa.text('now()')),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("action", sa.String(), nullable=True),
        sa.Column("edited_text", sa.Text(), nullable=True),
        sa.Column("score", sa.Integer(), nullable=True, server_default="0"),
        sa.Column("completed", sa.Boolean(), nullable=False, server_default=sa.text('false')),
        sa.UniqueConstraint("user_id", "base", name="uq_user_task"),
    )

def downgrade():
    op.drop_table("file_assignments")
    op.drop_table("users")
