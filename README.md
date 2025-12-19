# RYTHUVAANI_MAIN_BACKEND

## Overview

This repository contains the backend service for the **Rythuvaani Audio–Text Verification Platform**. The backend is built using **Python (FastAPI)** with **PostgreSQL** as the database and **AWS S3** as the storage layer for media and metadata files.

This README is a **complete rewrite of the previous developer README**, preserving all original intent while clearly documenting **all structural, architectural, and functional changes made during the current development cycle**.

The system supports:
- User registration and Google-based login
- Audio–text verification workflows (Correct / Edit / Incorrect)
- Metadata management at JSON level
- Admin dashboards, summaries, and task assignment
- Scalable task distribution from AWS S3

---

## Major Changes From Previous Backend

The following changes were implemented compared to the earlier version:

### 1. Storage Migration
- **Old**: Azure Blob Storage
- **New**: AWS S3 (fully replaced)

All `.wav`, `.txt`, and `.json` files are now:
- Listed from AWS S3
- Read using async S3 clients
- Written back to S3 on edit / incorrect flows

There is **no Azure dependency left** in the system.

---

### 2. Metadata Support (New Feature)

Previously:
- No concept of JSON metadata
- Only audio and text files were handled

Now:
- Each task can have an associated **JSON metadata file**
- Metadata is **updated in-place** when:
  - User marks a task as `correct`
  - User edits text (`edit` action)
- Metadata fields supported:
  - `Gender`
  - `age`
  - `Dialect`
  - `Mode`
  - `edited_text` (only for edit action)

Important behavior:
- Original text is preserved
- Edited text is stored separately (`edited_text` field)
- Metadata JSON is updated **only if the main JSON already exists** (strict behavior)

---

### 3. Incorrect Flow Enhancement

Previously:
- Only `.wav` and `.txt` files were moved to `incorrect/`

Now:
- Audio file
- Text file
- Corresponding JSON metadata entry

are all moved/updated consistently under the `incorrect/` prefix.

---

### 4. New Admin APIs Added

The following **new admin endpoints** were introduced:

#### Admin Overview
Provides system-wide task statistics:
- Total tasks in S3
- Assigned tasks
- Unassigned tasks
- Correct / Incorrect / Edited counts

```
GET /admin/overview
```

---

#### Delete User (Admin Only)
Deletes a user **and all assigned tasks** safely.

```
DELETE /admin/user/{user_id}
```

---

#### Assign Tasks to User (Admin Only)
Allows admin to manually assign unassigned tasks to a specific user.

```
POST /admin/tasks/assign
```

Payload:
```
{
  "user_id": 12,
  "limit": 10
}
```

---

### 5. User Tasks API – Access Change

Previously:
- `/user/{user_id}/tasks` required admin privileges

Now:
- **Admin restriction removed**
- Users can fetch **their own assigned tasks**

This endpoint now:
- Reads text from S3
- Generates pre-signed audio URLs
- Returns task status, score, and completion metadata

---

## Task Allocation Configuration (IMPORTANT)

### 1. Number of Tasks Assigned Per User

Controlled via environment variable:

```
TASKS_PER_USER=50
```

Location:
- `.env` file
- Used during initial task assignment on first login

---

### 2. AWS S3 Folder & Path Configuration

Key configuration values:

```
S3_BUCKET=your-bucket-name
AWS_REGION=ap-south-1
DEFAULT_TESTING_PREFIX=Testingsakeeth/
```

Behavior:
- Audio files: `.wav`, `.mp3`, etc.
- Text files: `.txt`
- Metadata JSON: `<folder>.json`

Multiple folders are supported automatically via S3 prefix scanning.

---

## Project Structure (Current)

```
RYTHUVAANI_MAIN_BACKEND/
├── alembic/
│   ├── versions/
│   │   ├── 0001_initial_create_tables.py
│   │   └── 0002_add_gender_age_dialect_and_s3_columns.py
│   └── env.py
├── venv/
├── .env
├── alembic.ini
├── app.py
├── requirements.txt
└── README.md
```

Notes:
- Alembic is used for schema migrations
- Async SQLAlchemy engine is used
- No modular split yet (single `app.py` by design)

---

## Database Schema (High Level)

### User Table
- id
- full_name
- email
- college_id
- college_name
- mobile_number
- gender
- age
- dialect
- mode
- is_admin
- session_start
- session_end

### FileAssignment Table
- id
- user_id (FK)
- base
- audio_key
- text_key
- json_key
- action (correct / incorrect / edit)
- edited_text
- score
- completed
- timestamps

Unique constraint:
```
(user_id, base)
```

---

## Tech Stack

- **Backend**: FastAPI (Async)
- **Database**: PostgreSQL
- **ORM**: SQLAlchemy (Async)
- **Migrations**: Alembic
- **Auth**: JWT + Google OAuth
- **Storage**: AWS S3
- **Server**: Uvicorn

---

## Environment Variables (Required)

```
DATABASE_URL=postgresql+asyncpg://...
SECRET_KEY=...
GOOGLE_CLIENT_ID=...
ADMIN_EMAIL=...
ADMIN_PASSWORD=...
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=ap-south-1
S3_BUCKET=...
TASKS_PER_USER=50
```

---

## Running the Project

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
```

Swagger:
```
http://127.0.0.1:8000/docs
```

---

## Summary

This backend is now:
- Fully AWS S3–based
- Metadata-aware
- Admin-operable with task reassignment
- Scalable and async-safe
- Backward compatible in intent, but significantly improved in design

This README supersedes all previous documentation and reflects the **current authoritative backend state**.

