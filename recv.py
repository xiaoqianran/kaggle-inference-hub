import uvicorn

from hub.app import app
from hub.config import PORT


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
