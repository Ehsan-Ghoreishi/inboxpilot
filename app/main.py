from fastapi import FastAPI

from app.api.events import router as events_router
from app.api.health import router as health_router

app = FastAPI(title="InboxPilot")
app.include_router(health_router)
app.include_router(events_router)


def main() -> None:
    print("Hello, InboxPilot")


if __name__ == "__main__":
    main()
