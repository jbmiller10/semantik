"""Internal API endpoints for system services"""

from fastapi import APIRouter

from .. import database

router = APIRouter(prefix="/api/internal", tags=["internal"])


@router.get("/jobs/all-ids")
def get_all_job_ids() -> list[str]:
    """
    Get all job IDs from the database.
    This endpoint is intended for internal services like maintenance/cleanup.
    """
    jobs = database.list_jobs()
    return [job["id"] for job in jobs]
