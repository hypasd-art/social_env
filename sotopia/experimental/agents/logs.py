from typing import Any

from redis_om import JsonModel
from sotopia.database.logs import BaseEpisodeLog
from sotopia.database.persistent_profile import AgentProfile


class EpisodeLog(BaseEpisodeLog, JsonModel):
    def __init__(self, **kwargs: Any):
        if "pk" not in kwargs:
            kwargs["pk"] = ""
        super().__init__(**kwargs)

    def render_for_humans(self) -> tuple[list[AgentProfile], list[str]]:
        raise NotImplementedError
