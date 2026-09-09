"""Read-only acquisition and faithful normalization of a visible Jira project."""

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import re

import httpx

from app.graph.models import Edge, GraphScope, Issue


class JiraCaptureError(ValueError):
    """Safe acquisition error; never contains response bodies or credentials."""


@dataclass
class CapturedProject:
    issues: list[dict]
    started_at: datetime
    finished_at: datetime
    account_scope: str
    diagnostics: list[str]


@dataclass
class NormalizedProject:
    issues: list[Issue]
    edges: list[Edge]
    texts: dict[str, str]
    diagnostics: list[str]


class JiraSnapshotReader:
    def __init__(self, client: httpx.Client, site_url: str, allowed_projects: tuple[str, ...]):
        self.client = client
        self.site_url = site_url.rstrip("/")
        self.allowed_projects = allowed_projects

    def _get(self, path: str, params: dict | None = None) -> dict:
        try:
            response = self.client.get(self.site_url + path, params=params)
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as exc:
            raise JiraCaptureError(f"Jira request failed: HTTP {exc.response.status_code}") from None
        except (httpx.RequestError, ValueError):
            raise JiraCaptureError("Jira request failed or response was invalid") from None
        if not isinstance(data, dict):
            raise JiraCaptureError("Jira returned an invalid object")
        return data

    def capture(self, project_key: str) -> CapturedProject:
        if project_key not in self.allowed_projects or not re.fullmatch(r"[A-Z][A-Z0-9_]*", project_key):
            raise JiraCaptureError("Project is not allowed")
        started = datetime.now(timezone.utc)
        account = self._get("/rest/api/3/myself").get("accountId")
        if not isinstance(account, str) or not account:
            raise JiraCaptureError("Jira account scope unavailable")
        account_scope = sha256(f"{self.site_url}\0{account}".encode()).hexdigest()
        params = {
            "jql": f'project = "{project_key}" ORDER BY id ASC',
            "maxResults": 100,
            "fields": "project,issuetype,summary,description,status,parent,issuelinks,updated",
        }
        seen_tokens = set()
        records = {}
        while True:
            page = self._get("/rest/api/3/search/jql", params)
            if not isinstance(page.get("issues"), list) or type(page.get("isLast")) is not bool:
                raise JiraCaptureError("Jira pagination completeness unavailable")
            for record in page["issues"]:
                if not isinstance(record, dict):
                    raise JiraCaptureError("Jira issue record invalid")
                fields = record.get("fields")
                identifier = record.get("id")
                if (not isinstance(fields, dict) or not isinstance(identifier, str) or not identifier
                        or not isinstance(fields.get("project"), dict)
                        or fields.get("project", {}).get("key") != project_key):
                    raise JiraCaptureError("Jira issue identity or project boundary invalid")
                if identifier in records and records[identifier] != record:
                    raise JiraCaptureError("Jira issue changed during capture")
                records[identifier] = record
            if page["isLast"]:
                break
            token = page.get("nextPageToken")
            if not isinstance(token, str) or not token or token in seen_tokens:
                raise JiraCaptureError("Jira pagination token missing or repeated")
            seen_tokens.add(token)
            params["nextPageToken"] = token
        return CapturedProject(list(records.values()), started, datetime.now(timezone.utc),
                               account_scope, [])


def _text(node) -> str:
    if isinstance(node, str):
        return node
    if not isinstance(node, dict):
        return ""
    kind = node.get("type")
    if kind == "text":
        return node.get("text", "")
    if kind == "hardBreak":
        return "\n"
    text = "".join(_text(child) for child in node.get("content", []))
    if kind in {"paragraph", "heading", "listItem", "codeBlock", "blockquote", "tableRow"}:
        text += "\n"
    return text


def normalize(scope: GraphScope, capture: CapturedProject, site_url: str) -> NormalizedProject:
    issues, edges, texts = [], [], {}
    diagnostics = list(capture.diagnostics)
    records = {}
    for record in capture.issues:
        if record["fields"]["project"]["key"] != scope.project_key:
            raise JiraCaptureError("Jira normalization project boundary invalid")
        if record["id"] in records and records[record["id"]] != record:
            raise JiraCaptureError("Jira normalization duplicate identity")
        records[record["id"]] = record
    seen = set()
    for identifier, record in records.items():
        fields = record["fields"]
        status = fields["status"]
        title = fields.get("summary") or ""
        issues.append(Issue(
            scope=scope, issue_id=identifier, key=record["key"],
            issue_type=fields["issuetype"]["name"], title=title,
            status=status["name"], status_category=status["statusCategory"]["key"],
            source_url=f"{site_url.rstrip('/')}/browse/{record['key']}",
            updated_at=datetime.fromisoformat(fields["updated"]),
            document_id="jira:" + sha256(f"{scope.site_id}\0{identifier}".encode()).hexdigest(),
        ))
        texts[identifier] = "\n\n".join(part for part in (title, _text(fields.get("description")).strip()) if part)
        parent = fields.get("parent")
        if parent:
            target = parent.get("id")
            if target in records:
                edges.append(Edge(scope=scope, source_issue_id=identifier, target_issue_id=target,
                                  relation_type="CHILD_OF", origin_issue_id=identifier,
                                  source_field="parent", source_type="parent", source_direction="parent"))
            else:
                diagnostics.append(f"boundary: parent of {identifier} is outside visible snapshot")
        elif fields["issuetype"]["name"].lower() != "epic":
            diagnostics.append(f"unassigned: issue {identifier} has no parent")
        for link in fields.get("issuelinks", []):
            raw_type = link.get("type", {})
            name = raw_type.get("name") or "Unknown"
            direction = "outward" if "outwardIssue" in link else "inward"
            target = link.get(direction + "Issue", {}).get("id")
            if target not in records:
                diagnostics.append(f"boundary: issue link from {identifier} is outside visible snapshot")
                continue
            source, destination = (identifier, target) if direction == "outward" else (target, identifier)
            relation = {"blocks": "BLOCKS", "relates": "RELATED_TO", "relates to": "RELATED_TO"}.get(name.lower(), "UNKNOWN")
            # Relates is symmetric; canonicalize endpoints for reciprocal records.
            if relation == "RELATED_TO":
                source, destination = sorted((source, destination))
            identity = (source, destination, relation, name, link.get("id"))
            if identity in seen:
                continue
            seen.add(identity)
            edges.append(Edge(scope=scope, source_issue_id=source, target_issue_id=destination,
                              relation_type=relation, origin_issue_id=identifier,
                              source_field="issuelinks", source_type=name,
                              source_direction=direction, link_id=link.get("id")))
    return NormalizedProject(issues, edges, texts, diagnostics)
