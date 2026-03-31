from atlassian import Jira

from app.config import settings
from app.connectors.base import BaseConnector, Document


class JiraConnector(BaseConnector):
    def __init__(self):
        self.client = Jira(
            url=settings.jira_url,
            username=settings.jira_email,
            password=settings.jira_api_token,
            cloud=True,
        )

    def fetch(self, project_key: str, max_results: int = 100, **kwargs) -> list[Document]:
        jql = f"project = {project_key} ORDER BY created DESC"
        issues = self.client.jql(jql, limit=max_results)["issues"]
        documents = []
        for issue in issues:
            fields = issue["fields"]
            summary = fields.get("summary", "")
            description = fields.get("description") or ""
            content = f"Summary: {summary}\n\nDescription: {description}"
            comments = fields.get("comment", {}).get("comments", [])
            if comments:
                comment_lines = [
                    f"Comment by {c['author']['displayName']}: {c['body']}"
                    for c in comments
                ]
                content += "\n\nComments:\n" + "\n".join(comment_lines)
            documents.append(
                Document(
                    id=self.make_id("jira", issue["key"]),
                    content=content,
                    source_type="jira",
                    title=f"[{issue['key']}] {summary}",
                    metadata={
                        "issue_key": issue["key"],
                        "status": fields.get("status", {}).get("name", ""),
                        "priority": fields.get("priority", {}).get("name", "") if fields.get("priority") else "",
                        "reporter": fields.get("reporter", {}).get("displayName", "") if fields.get("reporter") else "",
                        "created_at": fields.get("created", ""),
                    },
                )
            )
        return documents
