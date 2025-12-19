"""add gender/age/dialect to users and add/rename s3 key columns

Revision ID: 0002_add_gender_age_dialect_and_s3_columns
Revises: 0001_initial
Create Date: 2025-02-16 00:30:00
"""
from alembic import op
import sqlalchemy as sa

revision = "0002_add_gender_age_dialect_and_s3_columns"
down_revision = "0001_initial"
branch_labels = None
depends_on = None

def column_exists(connection, table_name, column_name):
    q = sa.text("""
        SELECT 1 FROM information_schema.columns
        WHERE table_name = :table AND column_name = :column
        LIMIT 1
    """)
    result = connection.execute(q, {"table": table_name, "column": column_name}).fetchone()
    return result is not None

def upgrade():
    conn = op.get_bind()

    # --- USERS: add gender, age, dialect if missing ---
    if not column_exists(conn, "users", "gender"):
        op.add_column("users", sa.Column("gender", sa.String(), nullable=True))
    if not column_exists(conn, "users", "age"):
        op.add_column("users", sa.Column("age", sa.Integer(), nullable=True))
    if not column_exists(conn, "users", "dialect"):
        op.add_column("users", sa.Column("dialect", sa.String(), nullable=True))

    # --- FILE_ASSIGNMENTS: rename audio_blob->audio_key if present, add text/audio/json keys if missing ---
    # rename audio_blob -> audio_key
    if column_exists(conn, "file_assignments", "audio_blob") and not column_exists(conn, "file_assignments", "audio_key"):
        # Use batch_alter_table to be safe with sqlite/postgres
        with op.batch_alter_table("file_assignments") as batch:
            batch.alter_column("audio_blob", new_column_name="audio_key")
    # rename text_blob -> text_key
    if column_exists(conn, "file_assignments", "text_blob") and not column_exists(conn, "file_assignments", "text_key"):
        with op.batch_alter_table("file_assignments") as batch:
            batch.alter_column("text_blob", new_column_name="text_key")

    # add audio_key/text_key/json_key if missing (may already have been created by rename)
    if not column_exists(conn, "file_assignments", "audio_key"):
        op.add_column("file_assignments", sa.Column("audio_key", sa.String(), nullable=False))
    if not column_exists(conn, "file_assignments", "text_key"):
        op.add_column("file_assignments", sa.Column("text_key", sa.String(), nullable=False))
    if not column_exists(conn, "file_assignments", "json_key"):
        op.add_column("file_assignments", sa.Column("json_key", sa.String(), nullable=True))


def downgrade():
    conn = op.get_bind()

    # drop json_key if exists
    if column_exists(conn, "file_assignments", "json_key"):
        with op.batch_alter_table("file_assignments") as batch:
            batch.drop_column("json_key")

    # attempt to rename audio_key back to audio_blob (only if audio_blob doesn't exist)
    if column_exists(conn, "file_assignments", "audio_key") and not column_exists(conn, "file_assignments", "audio_blob"):
        with op.batch_alter_table("file_assignments") as batch:
            batch.alter_column("audio_key", new_column_name="audio_blob")

    # attempt to rename text_key back to text_blob
    if column_exists(conn, "file_assignments", "text_key") and not column_exists(conn, "file_assignments", "text_blob"):
        with op.batch_alter_table("file_assignments") as batch:
            batch.alter_column("text_key", new_column_name="text_blob")

    # drop users columns if exist
    if column_exists(conn, "users", "dialect"):
        op.drop_column("users", "dialect")
    if column_exists(conn, "users", "age"):
        op.drop_column("users", "age")
    if column_exists(conn, "users", "gender"):
        op.drop_column("users", "gender")
