"""Integration tests for the v2 MCP profiles API."""

import os

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio()
async def test_create_profile_success(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    test_user_db,
    collection_factory,
) -> None:
    """Creating an MCP profile should persist and return the profile."""
    collection = await collection_factory(owner_id=test_user_db.id)

    response = await api_client.post(
        "/api/v2/mcp/profiles",
        headers=api_auth_headers,
        json={
            "name": "coding",
            "description": "Search coding documentation",
            "collection_ids": [collection.id],
            "enabled": True,
            "search_type": "semantic",
            "result_count": 10,
            "use_reranker": True,
        },
    )

    assert response.status_code == 201, response.text
    payload = response.json()
    assert payload["name"] == "coding"
    assert payload["description"] == "Search coding documentation"
    assert payload["enabled"] is True
    assert payload["search_type"] == "semantic"
    assert payload["result_count"] == 10
    assert len(payload["collections"]) == 1
    assert payload["collections"][0]["id"] == collection.id


@pytest.mark.asyncio()
async def test_create_profile_duplicate_name_fails(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    test_user_db,
    collection_factory,
) -> None:
    """Creating a profile with duplicate name should return 409."""
    collection = await collection_factory(owner_id=test_user_db.id)

    # Create first profile
    response = await api_client.post(
        "/api/v2/mcp/profiles",
        headers=api_auth_headers,
        json={
            "name": "coding",
            "description": "First profile",
            "collection_ids": [collection.id],
        },
    )
    assert response.status_code == 201, response.text

    # Try to create duplicate
    response = await api_client.post(
        "/api/v2/mcp/profiles",
        headers=api_auth_headers,
        json={
            "name": "coding",
            "description": "Duplicate profile",
            "collection_ids": [collection.id],
        },
    )
    assert response.status_code == 409, response.text


@pytest.mark.asyncio()
async def test_create_profile_invalid_name_fails(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    test_user_db,
    collection_factory,
) -> None:
    """Creating a profile with invalid name should return 422."""
    collection = await collection_factory(owner_id=test_user_db.id)

    # Name with uppercase
    response = await api_client.post(
        "/api/v2/mcp/profiles",
        headers=api_auth_headers,
        json={
            "name": "Coding",  # Invalid: uppercase
            "description": "Test profile",
            "collection_ids": [collection.id],
        },
    )
    assert response.status_code == 422, response.text


@pytest.mark.asyncio()
async def test_create_profile_nonexistent_collection_fails(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
) -> None:
    """Creating a profile with nonexistent collection should return 404."""
    response = await api_client.post(
        "/api/v2/mcp/profiles",
        headers=api_auth_headers,
        json={
            "name": "coding",
            "description": "Test profile",
            "collection_ids": ["00000000-0000-0000-0000-000000000000"],  # Valid UUID format, but doesn't exist
        },
    )
    assert response.status_code == 404, response.text


@pytest.mark.asyncio()
async def test_create_profile_invalid_uuid_format_fails(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
) -> None:
    """Creating a profile with invalid UUID format should return 422."""
    response = await api_client.post(
        "/api/v2/mcp/profiles",
        headers=api_auth_headers,
        json={
            "name": "coding",
            "description": "Test profile",
            "collection_ids": ["not-a-valid-uuid"],
        },
    )
    assert response.status_code == 422, response.text
    assert "Invalid UUID" in response.text


@pytest.mark.asyncio()
async def test_list_profiles_returns_owned_profiles(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    test_user_db,
    collection_factory,
) -> None:
    """Listing profiles should return all profiles owned by the user."""
    collection = await collection_factory(owner_id=test_user_db.id)

    # Create two profiles
    for name in ["coding", "personal"]:
        response = await api_client.post(
            "/api/v2/mcp/profiles",
            headers=api_auth_headers,
            json={
                "name": name,
                "description": f"{name} profile",
                "collection_ids": [collection.id],
            },
        )
        assert response.status_code == 201, response.text

    response = await api_client.get("/api/v2/mcp/profiles", headers=api_auth_headers)

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["total"] >= 2
    names = {p["name"] for p in payload["profiles"]}
    assert "coding" in names
    assert "personal" in names


@pytest.mark.asyncio()
async def test_list_profiles_filter_by_enabled(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    test_user_db,
    collection_factory,
) -> None:
    """Listing profiles with enabled filter should return matching profiles."""
    collection = await collection_factory(owner_id=test_user_db.id)

    # Create enabled profile
    await api_client.post(
        "/api/v2/mcp/profiles",
        headers=api_auth_headers,
        json={
            "name": "enabled-profile",
            "description": "Enabled",
            "collection_ids": [collection.id],
            "enabled": True,
        },
    )

    # Create disabled profile
    await api_client.post(
        "/api/v2/mcp/profiles",
        headers=api_auth_headers,
        json={
            "name": "disabled-profile",
            "description": "Disabled",
            "collection_ids": [collection.id],
            "enabled": False,
        },
    )

    # Filter by enabled=true
    response = await api_client.get("/api/v2/mcp/profiles?enabled=true", headers=api_auth_headers)
    assert response.status_code == 200, response.text
    payload = response.json()
    assert all(p["enabled"] for p in payload["profiles"])


@pytest.mark.asyncio()
async def test_get_profile_success(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    test_user_db,
    collection_factory,
) -> None:
    """Getting a profile should return its full details."""
    collection = await collection_factory(owner_id=test_user_db.id)

    create_response = await api_client.post(
        "/api/v2/mcp/profiles",
        headers=api_auth_headers,
        json={
            "name": "coding",
            "description": "Search coding documentation",
            "collection_ids": [collection.id],
        },
    )
    profile_id = create_response.json()["id"]

    response = await api_client.get(f"/api/v2/mcp/profiles/{profile_id}", headers=api_auth_headers)

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["id"] == profile_id
    assert payload["name"] == "coding"


@pytest.mark.asyncio()
async def test_get_profile_not_found(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
) -> None:
    """Getting a nonexistent profile should return 404."""
    response = await api_client.get("/api/v2/mcp/profiles/nonexistent-id", headers=api_auth_headers)
    assert response.status_code == 404, response.text


@pytest.mark.asyncio()
async def test_update_profile_success(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    test_user_db,
    collection_factory,
) -> None:
    """Updating a profile should persist changes."""
    collection = await collection_factory(owner_id=test_user_db.id)

    create_response = await api_client.post(
        "/api/v2/mcp/profiles",
        headers=api_auth_headers,
        json={
            "name": "coding",
            "description": "Original description",
            "collection_ids": [collection.id],
            "result_count": 10,
        },
    )
    profile_id = create_response.json()["id"]

    # Update the profile
    response = await api_client.put(
        f"/api/v2/mcp/profiles/{profile_id}",
        headers=api_auth_headers,
        json={
            "description": "Updated description",
            "result_count": 20,
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["description"] == "Updated description"
    assert payload["result_count"] == 20
    # Name should remain unchanged
    assert payload["name"] == "coding"


@pytest.mark.asyncio()
async def test_update_profile_name_conflict(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    test_user_db,
    collection_factory,
) -> None:
    """Updating profile name to existing name should return 409."""
    collection = await collection_factory(owner_id=test_user_db.id)

    # Create two profiles
    await api_client.post(
        "/api/v2/mcp/profiles",
        headers=api_auth_headers,
        json={
            "name": "first",
            "description": "First",
            "collection_ids": [collection.id],
        },
    )

    second_response = await api_client.post(
        "/api/v2/mcp/profiles",
        headers=api_auth_headers,
        json={
            "name": "second",
            "description": "Second",
            "collection_ids": [collection.id],
        },
    )
    second_id = second_response.json()["id"]

    # Try to rename second to first
    response = await api_client.put(
        f"/api/v2/mcp/profiles/{second_id}",
        headers=api_auth_headers,
        json={"name": "first"},
    )
    assert response.status_code == 409, response.text


@pytest.mark.asyncio()
async def test_delete_profile_success(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    test_user_db,
    collection_factory,
) -> None:
    """Deleting a profile should remove it."""
    collection = await collection_factory(owner_id=test_user_db.id)

    create_response = await api_client.post(
        "/api/v2/mcp/profiles",
        headers=api_auth_headers,
        json={
            "name": "coding",
            "description": "Test",
            "collection_ids": [collection.id],
        },
    )
    profile_id = create_response.json()["id"]

    # Delete the profile
    response = await api_client.delete(f"/api/v2/mcp/profiles/{profile_id}", headers=api_auth_headers)
    assert response.status_code == 204, response.text

    # Verify it's gone
    response = await api_client.get(f"/api/v2/mcp/profiles/{profile_id}", headers=api_auth_headers)
    assert response.status_code == 404, response.text


@pytest.mark.asyncio()
async def test_delete_profile_not_found(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
) -> None:
    """Deleting a nonexistent profile should return 404."""
    response = await api_client.delete("/api/v2/mcp/profiles/nonexistent-id", headers=api_auth_headers)
    assert response.status_code == 404, response.text


@pytest.mark.asyncio()
async def test_get_profile_config(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    test_user_db,
    collection_factory,
) -> None:
    """Getting profile config should return MCP client configuration."""
    collection = await collection_factory(owner_id=test_user_db.id)

    create_response = await api_client.post(
        "/api/v2/mcp/profiles",
        headers=api_auth_headers,
        json={
            "name": "coding",
            "description": "Test",
            "collection_ids": [collection.id],
        },
    )
    profile_id = create_response.json()["id"]

    response = await api_client.get(f"/api/v2/mcp/profiles/{profile_id}/config", headers=api_auth_headers)

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["server_name"] == "semantik-coding"
    assert payload["command"] == "semantik-mcp"
    assert "--profile" in payload["args"]
    assert "coding" in payload["args"]
    assert "SEMANTIK_WEBUI_URL" in payload["env"]
    assert "SEMANTIK_AUTH_TOKEN" in payload["env"]


@pytest.mark.asyncio()
@pytest.mark.skipif(
    os.environ.get("DISABLE_AUTH", "").lower() in ("true", "1", "yes"),
    reason="Auth is disabled in test environment",
)
async def test_auth_required(
    api_client_unauthenticated: AsyncClient,
) -> None:
    """All MCP profile endpoints should require authentication."""
    endpoints = [
        ("GET", "/api/v2/mcp/profiles"),
        ("POST", "/api/v2/mcp/profiles"),
        ("GET", "/api/v2/mcp/profiles/some-id"),
        ("PUT", "/api/v2/mcp/profiles/some-id"),
        ("DELETE", "/api/v2/mcp/profiles/some-id"),
        ("GET", "/api/v2/mcp/profiles/some-id/config"),
    ]

    for method, path in endpoints:
        if method == "GET":
            response = await api_client_unauthenticated.get(path)
        elif method == "POST":
            response = await api_client_unauthenticated.post(path, json={})
        elif method == "PUT":
            response = await api_client_unauthenticated.put(path, json={})
        elif method == "DELETE":
            response = await api_client_unauthenticated.delete(path)

        assert response.status_code == 401, f"{method} {path} should require auth: {response.text}"


@pytest.mark.asyncio()
async def test_profile_multiple_collections_ordering(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    test_user_db,
    collection_factory,
) -> None:
    """Profile should preserve collection ordering."""
    collection1 = await collection_factory(owner_id=test_user_db.id)
    collection2 = await collection_factory(owner_id=test_user_db.id)
    collection3 = await collection_factory(owner_id=test_user_db.id)

    # Create profile with specific order
    response = await api_client.post(
        "/api/v2/mcp/profiles",
        headers=api_auth_headers,
        json={
            "name": "multi",
            "description": "Multiple collections",
            "collection_ids": [collection3.id, collection1.id, collection2.id],
        },
    )

    assert response.status_code == 201, response.text
    payload = response.json()
    collection_ids = [c["id"] for c in payload["collections"]]
    # Order should be preserved
    assert collection_ids == [collection3.id, collection1.id, collection2.id]


@pytest.mark.asyncio()
async def test_profile_search_type_options(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    test_user_db,
    collection_factory,
) -> None:
    """Profile should accept all valid search types."""
    collection = await collection_factory(owner_id=test_user_db.id)

    for search_type in ["semantic", "hybrid", "keyword", "question", "code"]:
        response = await api_client.post(
            "/api/v2/mcp/profiles",
            headers=api_auth_headers,
            json={
                "name": f"profile-{search_type}",
                "description": f"Test {search_type}",
                "collection_ids": [collection.id],
                "search_type": search_type,
            },
        )
        assert response.status_code == 201, f"Failed for {search_type}: {response.text}"
        assert response.json()["search_type"] == search_type


# =============================================================================
# 403 Forbidden - Cross-user isolation tests
# =============================================================================


@pytest.mark.asyncio()
@pytest.mark.usefixtures("test_user_db")
async def test_get_profile_owned_by_other_user_returns_403(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    other_user_db,
    collection_factory,
    db_session,
) -> None:
    """User A cannot view User B's profile - returns 403 Forbidden."""
    from uuid import uuid4

    from shared.database.models import MCPProfile, MCPProfileCollection

    # Create a collection owned by other_user
    other_collection = await collection_factory(owner_id=other_user_db.id)

    # Directly create a profile owned by other_user in the database
    profile_id = str(uuid4())
    profile = MCPProfile(
        id=profile_id,
        name="other-user-profile",
        description="Profile owned by other user",
        owner_id=other_user_db.id,
        enabled=True,
        search_type="semantic",
        result_count=10,
        use_reranker=True,
    )
    db_session.add(profile)

    assoc = MCPProfileCollection(
        profile_id=profile_id,
        collection_id=other_collection.id,
        order=0,
    )
    db_session.add(assoc)
    await db_session.flush()

    # test_user tries to access other_user's profile - should get 403
    response = await api_client.get(
        f"/api/v2/mcp/profiles/{profile_id}",
        headers=api_auth_headers,
    )
    assert response.status_code == 403, f"Expected 403, got {response.status_code}: {response.text}"


@pytest.mark.asyncio()
@pytest.mark.usefixtures("test_user_db")
async def test_update_profile_owned_by_other_user_returns_403(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    other_user_db,
    collection_factory,
    db_session,
) -> None:
    """User A cannot update User B's profile - returns 403 Forbidden."""
    from uuid import uuid4

    from shared.database.models import MCPProfile, MCPProfileCollection

    # Create a collection owned by other_user
    other_collection = await collection_factory(owner_id=other_user_db.id)

    # Directly create a profile owned by other_user in the database
    profile_id = str(uuid4())
    profile = MCPProfile(
        id=profile_id,
        name="other-user-profile",
        description="Profile owned by other user",
        owner_id=other_user_db.id,
        enabled=True,
        search_type="semantic",
        result_count=10,
        use_reranker=True,
    )
    db_session.add(profile)

    assoc = MCPProfileCollection(
        profile_id=profile_id,
        collection_id=other_collection.id,
        order=0,
    )
    db_session.add(assoc)
    await db_session.flush()

    # test_user tries to update other_user's profile - should get 403
    response = await api_client.put(
        f"/api/v2/mcp/profiles/{profile_id}",
        headers=api_auth_headers,
        json={"description": "Trying to update someone else's profile"},
    )
    assert response.status_code == 403, f"Expected 403, got {response.status_code}: {response.text}"


@pytest.mark.asyncio()
@pytest.mark.usefixtures("test_user_db")
async def test_delete_profile_owned_by_other_user_returns_403(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    other_user_db,
    collection_factory,
    db_session,
) -> None:
    """User A cannot delete User B's profile - returns 403 Forbidden."""
    from uuid import uuid4

    from shared.database.models import MCPProfile, MCPProfileCollection

    # Create a collection owned by other_user
    other_collection = await collection_factory(owner_id=other_user_db.id)

    # Directly create a profile owned by other_user in the database
    profile_id = str(uuid4())
    profile = MCPProfile(
        id=profile_id,
        name="other-user-profile",
        description="Profile owned by other user",
        owner_id=other_user_db.id,
        enabled=True,
        search_type="semantic",
        result_count=10,
        use_reranker=True,
    )
    db_session.add(profile)

    assoc = MCPProfileCollection(
        profile_id=profile_id,
        collection_id=other_collection.id,
        order=0,
    )
    db_session.add(assoc)
    await db_session.flush()

    # test_user tries to delete other_user's profile - should get 403
    response = await api_client.delete(
        f"/api/v2/mcp/profiles/{profile_id}",
        headers=api_auth_headers,
    )
    assert response.status_code == 403, f"Expected 403, got {response.status_code}: {response.text}"


@pytest.mark.asyncio()
@pytest.mark.usefixtures("test_user_db")
async def test_get_profile_config_owned_by_other_user_returns_403(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    other_user_db,
    collection_factory,
    db_session,
) -> None:
    """User A cannot get config for User B's profile - returns 403 Forbidden."""
    from uuid import uuid4

    from shared.database.models import MCPProfile, MCPProfileCollection

    # Create a collection owned by other_user
    other_collection = await collection_factory(owner_id=other_user_db.id)

    # Directly create a profile owned by other_user in the database
    profile_id = str(uuid4())
    profile = MCPProfile(
        id=profile_id,
        name="other-user-profile",
        description="Profile owned by other user",
        owner_id=other_user_db.id,
        enabled=True,
        search_type="semantic",
        result_count=10,
        use_reranker=True,
    )
    db_session.add(profile)

    assoc = MCPProfileCollection(
        profile_id=profile_id,
        collection_id=other_collection.id,
        order=0,
    )
    db_session.add(assoc)
    await db_session.flush()

    # test_user tries to get config for other_user's profile - should get 403
    response = await api_client.get(
        f"/api/v2/mcp/profiles/{profile_id}/config",
        headers=api_auth_headers,
    )
    assert response.status_code == 403, f"Expected 403, got {response.status_code}: {response.text}"


@pytest.mark.asyncio()
async def test_list_profiles_only_returns_owned_profiles(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    test_user_db,
    other_user_db,
    collection_factory,
    db_session,
) -> None:
    """List profiles should not return other users' profiles."""
    from uuid import uuid4

    from shared.database.models import MCPProfile, MCPProfileCollection

    # Create collections for each user
    user_collection = await collection_factory(owner_id=test_user_db.id)
    other_collection = await collection_factory(owner_id=other_user_db.id)

    # test_user creates a profile via API
    await api_client.post(
        "/api/v2/mcp/profiles",
        headers=api_auth_headers,
        json={
            "name": "my-profile",
            "description": "My profile",
            "collection_ids": [user_collection.id],
        },
    )

    # Directly create a profile owned by other_user in the database
    other_profile_id = str(uuid4())
    other_profile = MCPProfile(
        id=other_profile_id,
        name="other-profile",
        description="Other user's profile",
        owner_id=other_user_db.id,
        enabled=True,
        search_type="semantic",
        result_count=10,
        use_reranker=True,
    )
    db_session.add(other_profile)

    assoc = MCPProfileCollection(
        profile_id=other_profile_id,
        collection_id=other_collection.id,
        order=0,
    )
    db_session.add(assoc)
    await db_session.flush()

    # test_user lists profiles - should only see their own
    response = await api_client.get(
        "/api/v2/mcp/profiles",
        headers=api_auth_headers,
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    profile_names = {p["name"] for p in payload["profiles"]}
    assert "my-profile" in profile_names
    assert "other-profile" not in profile_names, "Should not see other user's profiles"


# =============================================================================
# 403 Forbidden - Collection access validation tests
# =============================================================================


@pytest.mark.asyncio()
@pytest.mark.usefixtures("test_user_db")
async def test_create_profile_with_unowned_collection_returns_403(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    other_user_db,
    collection_factory,
) -> None:
    """Cannot create profile with collection owned by another user - returns 403."""
    # Create a collection owned by other_user
    other_collection = await collection_factory(owner_id=other_user_db.id)

    # test_user tries to create a profile with other_user's collection
    response = await api_client.post(
        "/api/v2/mcp/profiles",
        headers=api_auth_headers,
        json={
            "name": "sneaky-profile",
            "description": "Trying to use someone else's collection",
            "collection_ids": [other_collection.id],
        },
    )
    assert response.status_code == 403, f"Expected 403, got {response.status_code}: {response.text}"


@pytest.mark.asyncio()
async def test_update_profile_to_add_unowned_collection_returns_403(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    test_user_db,
    other_user_db,
    collection_factory,
) -> None:
    """Cannot update profile to add collection owned by another user - returns 403."""
    # Create collections for each user
    user_collection = await collection_factory(owner_id=test_user_db.id)
    other_collection = await collection_factory(owner_id=other_user_db.id)

    # test_user creates a valid profile
    create_response = await api_client.post(
        "/api/v2/mcp/profiles",
        headers=api_auth_headers,
        json={
            "name": "my-profile",
            "description": "My profile",
            "collection_ids": [user_collection.id],
        },
    )
    assert create_response.status_code == 201, create_response.text
    profile_id = create_response.json()["id"]

    # test_user tries to update profile to add other_user's collection
    response = await api_client.put(
        f"/api/v2/mcp/profiles/{profile_id}",
        headers=api_auth_headers,
        json={
            "collection_ids": [user_collection.id, other_collection.id],
        },
    )
    assert response.status_code == 403, f"Expected 403, got {response.status_code}: {response.text}"


@pytest.mark.asyncio()
async def test_create_profile_with_mix_of_owned_and_unowned_collections_returns_403(
    api_client: AsyncClient,
    api_auth_headers: dict[str, str],
    test_user_db,
    other_user_db,
    collection_factory,
) -> None:
    """Cannot create profile with any unowned collection, even if some are owned."""
    # Create collections for each user
    user_collection = await collection_factory(owner_id=test_user_db.id)
    other_collection = await collection_factory(owner_id=other_user_db.id)

    # test_user tries to create a profile with both their own and other's collection
    response = await api_client.post(
        "/api/v2/mcp/profiles",
        headers=api_auth_headers,
        json={
            "name": "mixed-profile",
            "description": "Trying to use mix of collections",
            "collection_ids": [user_collection.id, other_collection.id],
        },
    )
    assert response.status_code == 403, f"Expected 403, got {response.status_code}: {response.text}"
