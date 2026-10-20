from mcp.server.mcpserver import MCPServer
import httpx

mcp = MCPServer("Learning MCP")

@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

@mcp.tool()
def get_github_user(username: str) -> dict:
    """Get public information about a GitHub user."""

    response = httpx.get(
        f"https://api.github.com/users/{username}",
        timeout=10
    )

    response.raise_for_status()
    data = response.json()

    return {
        "username": data["login"],
        "name": data["name"],
        "public_repos": data["public_repos"],
        "profile_url": data["html_url"]
    }

if __name__ == "__main__":
    mcp.run()