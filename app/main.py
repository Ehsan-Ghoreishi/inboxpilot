from fastapi import FastAPI

from app.api.actions import router as actions_router
from app.api.dashboard import router as dashboard_router
from app.api.events import router as events_router
from app.api.health import router as health_router
from app.api.search import router as search_router

app = FastAPI(title="InboxPilot")
app.include_router(health_router)
app.include_router(events_router)
app.include_router(actions_router)
app.include_router(dashboard_router)
app.include_router(search_router)


def main() -> None:
    print("Hello, InboxPilot")


if __name__ == "__main__":
    main()
