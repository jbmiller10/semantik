"""Internal API endpoints for system services"""

from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException
from shared.config import settings
from shared.database.base import JobRepository
from shared.database.factory import create_job_repository
from shared.database.compat import database  # TODO: Remove once tests are migrated

router = APIRouter(prefix="/api/internal", tags=["internal"])


def verify_internal_api_key(x_internal_api_key: Annotated[str | None, Header()] = None) -> None:
    """Verify the internal API key."""
    if x_internal_api_key != settings.INTERNAL_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing internal API key")


@router.get("/jobs/all-ids", dependencies=[Depends(verify_internal_api_key)])
async def get_all_job_ids(job_repo: JobRepository = Depends(create_job_repository)) -> list[str]:
    """
    Get all job IDs from the database.
    This endpoint is intended for internal services like maintenance/cleanup.
    """
    jobs = await job_repo.list_jobs()
    return [job["id"] for job in jobs]
