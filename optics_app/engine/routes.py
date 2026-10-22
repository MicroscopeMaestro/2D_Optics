"""Backward-compatible routed optical paths for branched instruments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional


VALID_PORTS = {"transmit", "reflect", "absorb", "emit", "collect", "pass"}


@dataclass
class RouteStep:
    component_uid: str
    port: str = "pass"
    direction: str = "forward"

    def __post_init__(self) -> None:
        self.component_uid = str(self.component_uid or "").strip()
        self.port = str(self.port or "pass").strip().lower()
        self.direction = str(self.direction or "forward").strip().lower()
        if self.port not in VALID_PORTS:
            raise ValueError(f"Unsupported optical port '{self.port}'.")
        if self.direction not in ("forward", "reverse"):
            raise ValueError("Route direction must be 'forward' or 'reverse'.")

    def to_dict(self) -> Dict[str, str]:
        return {
            "component_uid": self.component_uid,
            "port": self.port,
            "direction": self.direction,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RouteStep":
        return cls(
            component_uid=str(data.get("component_uid", "")),
            port=str(data.get("port", "pass")),
            direction=str(data.get("direction", "forward")),
        )


@dataclass
class OpticalRoute:
    name: str
    steps: List[RouteStep] = field(default_factory=list)
    color: str = "#FFFFFF"
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "steps": [step.to_dict() for step in self.steps],
            "color": self.color,
            "enabled": bool(self.enabled),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OpticalRoute":
        return cls(
            name=str(data.get("name", "route")),
            steps=[RouteStep.from_dict(item) for item in data.get("steps", [])],
            color=str(data.get("color", "#FFFFFF")),
            enabled=bool(data.get("enabled", True)),
        )


@dataclass
class OpticalRouteSet:
    excitation: OpticalRoute = field(default_factory=lambda: OpticalRoute("excitation", color="#E53935"))
    dump: OpticalRoute = field(default_factory=lambda: OpticalRoute("dump", color="#FB8C00"))
    epi_detection: OpticalRoute = field(default_factory=lambda: OpticalRoute("epi_detection", color="#43A047"))
    extra_routes: Dict[str, OpticalRoute] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "excitation": self.excitation.to_dict(),
            "dump": self.dump.to_dict(),
            "epi_detection": self.epi_detection.to_dict(),
            "extra_routes": {name: route.to_dict() for name, route in self.extra_routes.items()},
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "OpticalRouteSet":
        payload = data or {}
        return cls(
            excitation=OpticalRoute.from_dict(payload.get("excitation", {"name": "excitation", "color": "#E53935"})),
            dump=OpticalRoute.from_dict(payload.get("dump", {"name": "dump", "color": "#FB8C00"})),
            epi_detection=OpticalRoute.from_dict(payload.get("epi_detection", {"name": "epi_detection", "color": "#43A047"})),
            extra_routes={
                str(name): OpticalRoute.from_dict(route)
                for name, route in (payload.get("extra_routes") or {}).items()
            },
        )

    def all_routes(self) -> Iterable[OpticalRoute]:
        return (self.excitation, self.dump, self.epi_detection, *self.extra_routes.values())

    def validate(self, component_uids: Iterable[str]) -> List[str]:
        known = {str(uid) for uid in component_uids if uid}
        problems: List[str] = []
        for route in self.all_routes():
            for step in route.steps:
                if not step.component_uid:
                    problems.append(f"Route '{route.name}' contains an empty component reference.")
                elif step.component_uid not in known:
                    problems.append(
                        f"Route '{route.name}' references missing component '{step.component_uid}'."
                    )
        return problems

