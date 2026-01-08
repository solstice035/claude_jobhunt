from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.database import get_db
from app.models import Profile
from app.schemas import ProfileResponse, ProfileUpdate
from app.auth import get_current_user

router = APIRouter()

DEFAULT_PROFILE_ID = "default"


async def get_or_create_profile(db: AsyncSession) -> Profile:
    result = await db.execute(select(Profile).where(Profile.id == DEFAULT_PROFILE_ID))
    profile = result.scalar_one_or_none()

    if not profile:
        profile = Profile(id=DEFAULT_PROFILE_ID)
        db.add(profile)
        await db.commit()
        await db.refresh(profile)

    return profile


@router.get("", response_model=ProfileResponse)
async def get_profile(
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(get_current_user),
):
    profile = await get_or_create_profile(db)
    return ProfileResponse.model_validate(profile)


@router.put("", response_model=ProfileResponse)
async def update_profile(
    update: ProfileUpdate,
    db: AsyncSession = Depends(get_db),
    _: bool = Depends(get_current_user),
):
    profile = await get_or_create_profile(db)

    update_data = update.model_dump(exclude_unset=True)

    # Convert score_weights to dict if present
    if "score_weights" in update_data and update_data["score_weights"]:
        if hasattr(update_data["score_weights"], "model_dump"):
            update_data["score_weights"] = update_data["score_weights"].model_dump()

    cv_changed = "cv_text" in update_data and update_data["cv_text"] != profile.cv_text

    for field, value in update_data.items():
        setattr(profile, field, value)

    # Clear embedding if CV changed (will be regenerated on next match)
    if cv_changed:
        profile.cv_embedding = None

    await db.commit()
    await db.refresh(profile)

    return ProfileResponse.model_validate(profile)
